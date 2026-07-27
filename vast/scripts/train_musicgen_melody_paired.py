#!/usr/bin/env python3
"""
Smoke-train MusicGen Melody on paired melody -> instrumental chunks.

This is a small custom trainer for the project objective:

    input_audio + text -> target_audio

It uses AudioCraft directly:

1. Load `facebook/musicgen-melody`.
2. Encode `target_audio` into EnCodec tokens.
3. Build MusicGen melody conditioning from `input_audio`.
4. Train LM token prediction loss.

The default `--trainable linears` is intentionally conservative for smoke
testing. Use it to validate the pipeline before attempting heavier fine-tuning.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def train(args: argparse.Namespace) -> None:
    from audiocraft.data.audio_utils import convert_audio
    from audiocraft.models import MusicGen

    device = torch.device(args.device)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = MusicGen.get_pretrained(args.model, device=str(device))
    model.compression_model.eval()
    model.lm.train()

    _freeze(model.compression_model)
    trainable_names = _set_trainable(model.lm, args.trainable, args.last_n_layers)
    trainable_params = [
        param for param in model.lm.parameters() if param.requires_grad
    ]
    if not trainable_params:
        raise RuntimeError(f"No trainable parameters selected by: {args.trainable}")

    print(f"Model sample rate: {model.sample_rate}")
    print(f"Model channels:    {model.audio_channels}")
    print(f"Trainable mode:    {args.trainable}")
    print(f"Trainable tensors: {len(trainable_names)}")
    print(f"Trainable params:  {sum(p.numel() for p in trainable_params):,}")

    train_rows = _read_jsonl(args.train_metadata)
    valid_rows = _read_jsonl(args.valid_metadata) if args.valid_metadata else []
    if args.max_train_rows is not None:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_valid_rows is not None:
        valid_rows = valid_rows[: args.max_valid_rows]

    train_loader = DataLoader(
        PairDataset(train_rows, args.dataset_root),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: _collate(batch, model.sample_rate, model.audio_channels, convert_audio),
    )
    valid_loader = DataLoader(
        PairDataset(valid_rows, args.dataset_root),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: _collate(batch, model.sample_rate, model.audio_channels, convert_audio),
    ) if valid_rows else None

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    use_grad_scaler = (
        args.amp
        and device.type == "cuda"
        and all(param.dtype != torch.float16 for param in trainable_params)
    )
    if args.amp and not use_grad_scaler:
        print("GradScaler: disabled because trainable parameters are already FP16.")
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(train_loader, start=1):
            global_step += 1
            loss = _training_step(
                model=model,
                batch=batch,
                device=device,
                scaler=scaler,
                amp=args.amp,
            )
            loss_for_backward = loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if global_step % args.grad_accum_steps == 0:
                if args.grad_clip > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"[train] epoch={epoch + 1} step={global_step} "
                    f"loss={loss.item():.4f} elapsed={elapsed:.1f}s"
                )

            if valid_loader is not None and global_step % args.valid_every == 0:
                valid_loss = _evaluate(model, valid_loader, device, args.amp, args.max_valid_batches)
                print(f"[valid] step={global_step} loss={valid_loss:.4f}")

            if global_step % args.save_every == 0:
                _save_checkpoint(
                    output_dir / f"checkpoint_step_{global_step}.pt",
                    model=model,
                    model_name=args.model,
                    trainable_names=trainable_names,
                    global_step=global_step,
                    args=args,
                )

            if args.max_steps is not None and global_step >= args.max_steps:
                break

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    _save_checkpoint(
        output_dir / "checkpoint_last.pt",
        model=model,
        model_name=args.model,
        trainable_names=trainable_names,
        global_step=global_step,
        args=args,
    )
    print("Done.")
    print(f"Last checkpoint: {output_dir / 'checkpoint_last.pt'}")


class PairDataset(Dataset):
    def __init__(self, rows: list[dict], dataset_root: Path):
        self.rows = rows
        self.dataset_root = dataset_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {
            "input_audio": _resolve_path(row["input_audio"], self.dataset_root),
            "target_audio": _resolve_path(row["target_audio"], self.dataset_root),
            "text": row["text"],
            "track_id": row.get("track_id", ""),
            "chunk_index": row.get("chunk_index", -1),
        }


def _collate(batch: list[dict], sample_rate: int, channels: int, convert_audio_fn) -> dict:
    import soundfile as sf

    melodies = []
    targets = []
    texts = []
    meta = []

    for item in batch:
        melody, melody_sr = _load_wav(item["input_audio"], sf)
        target, target_sr = _load_wav(item["target_audio"], sf)

        melody = convert_audio_fn(melody, melody_sr, sample_rate, channels)
        target = convert_audio_fn(target, target_sr, sample_rate, channels)

        melodies.append(melody)
        targets.append(target)
        texts.append(item["text"])
        meta.append({"track_id": item["track_id"], "chunk_index": item["chunk_index"]})

    melodies = torch.stack(melodies, dim=0)
    targets = torch.stack(targets, dim=0)
    return {"melodies": melodies, "targets": targets, "texts": texts, "meta": meta}


def _load_wav(path: Path, sf_module) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf_module.read(str(path), dtype="float32", always_2d=True)
    tensor = torch.from_numpy(audio).transpose(0, 1).contiguous()
    return tensor, int(sample_rate)


def _training_step(model, batch: dict, device: torch.device, scaler, amp: bool) -> torch.Tensor:
    targets = batch["targets"].to(device)
    melodies = [wav.to(device) for wav in batch["melodies"]]
    texts = batch["texts"]

    with torch.no_grad():
        codes, scale = model.compression_model.encode(targets)
        if scale is not None:
            raise RuntimeError("Expected MusicGen compression scale to be None.")
        attributes, _ = model._prepare_tokens_and_attributes(
            descriptions=texts,
            prompt=None,
            melody_wavs=melodies,
        )

    autocast_enabled = amp and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
        output = model.lm.compute_predictions(codes, attributes)
        loss = _masked_cross_entropy(output.logits, codes, output.mask)
    return loss


@torch.no_grad()
def _evaluate(model, loader, device: torch.device, amp: bool, max_batches: int) -> float:
    was_training = model.lm.training
    model.lm.eval()
    losses = []
    for index, batch in enumerate(loader, start=1):
        loss = _training_step(model, batch, device, scaler=None, amp=amp)
        losses.append(float(loss.item()))
        if index >= max_batches:
            break
    if was_training:
        model.lm.train()
    return sum(losses) / max(1, len(losses))


def _masked_cross_entropy(logits: torch.Tensor, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_logits = logits[mask]
    valid_targets = codes[mask]
    if valid_logits.numel() == 0:
        raise RuntimeError("No valid logits for loss.")
    return F.cross_entropy(valid_logits, valid_targets)


def _freeze(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _set_trainable(lm: torch.nn.Module, mode: str, last_n_layers: int) -> list[str]:
    for param in lm.parameters():
        param.requires_grad = False

    if mode == "all":
        for param in lm.parameters():
            param.requires_grad = True
    elif mode == "linears":
        for module in lm.modules():
            if isinstance(module, torch.nn.Linear):
                for param in module.parameters():
                    param.requires_grad = True
    elif mode == "output_linears":
        if not hasattr(lm, "linears"):
            raise RuntimeError("LM has no `linears` module list.")
        for param in lm.linears.parameters():
            param.requires_grad = True
    elif mode == "last_layers":
        if not hasattr(lm, "transformer") or not hasattr(lm.transformer, "layers"):
            raise RuntimeError("LM transformer layers were not found.")
        layers = lm.transformer.layers
        if last_n_layers <= 0:
            raise ValueError("--last-n-layers must be > 0 for last_layers mode.")
        for layer in layers[-last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        if hasattr(lm, "linears"):
            for param in lm.linears.parameters():
                param.requires_grad = True
    else:
        raise ValueError(f"Unsupported trainable mode: {mode}")

    return [
        name
        for name, param in lm.named_parameters()
        if param.requires_grad
    ]


def _save_checkpoint(
    path: Path,
    model,
    model_name: str,
    trainable_names: list[str],
    global_step: int,
    args: argparse.Namespace,
) -> None:
    state = model.lm.state_dict()
    trainable = {
        name: state[name].detach().cpu()
        for name in trainable_names
        if name in state
    }
    torch.save(
        {
            "model_name": model_name,
            "global_step": global_step,
            "trainable": trainable,
            "trainable_names": trainable_names,
            "args": vars(args),
        },
        path,
    )
    print(f"[save] {path}")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
    return rows


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Custom paired smoke trainer for MusicGen Melody."
    )
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--valid-metadata", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="facebook/musicgen-melody-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--trainable",
        choices=["output_linears", "last_layers", "linears", "all"],
        default="output_linears",
    )
    parser.add_argument(
        "--last-n-layers",
        type=int,
        default=2,
        help="Number of final transformer layers to train when --trainable last_layers.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-valid-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--valid-every", type=int, default=25)
    parser.add_argument("--max-valid-batches", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    train(args)


if __name__ == "__main__":
    main()

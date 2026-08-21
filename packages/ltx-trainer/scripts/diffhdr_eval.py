#!/usr/bin/env python3
"""DiffHDR LDR→HDR video diffusion CLI (Yu et al. arXiv:2604.06161)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

import torch  # noqa: E402

from ltx_trainer.diffhdr.config import PAPER_TITLE, PAPER_URL, DiffHDRConfig  # noqa: E402
from ltx_trainer.diffhdr.dataset import dataset_summary  # noqa: E402
from ltx_trainer.diffhdr.model import DiffHDR  # noqa: E402
from ltx_trainer.diffhdr.pipeline import paper_report, save_checkpoint, train_step  # noqa: E402
from ltx_trainer.diffhdr.synthetic import synthesize_pair, vae_roundtrip_error  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    ds = dataset_summary()
    print(
        json.dumps(
            {
                "name": "DiffHDR",
                "paper": PAPER_TITLE,
                "url": PAPER_URL,
                "backbone": ds["backbone"],
                "modules": ["Log-Gamma mapping", "VACE DiT LoRA", "mask detection", "CFA", "flow matching"],
                "training_sequences": ds["stats"]["sequences"],
                "eval": list(ds["eval_datasets"].keys()),
            },
            indent=2,
        )
    )
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(paper_report(), indent=2))
    return 0


def _cmd_dataset(_: argparse.Namespace) -> int:
    print(json.dumps(dataset_summary(), indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = DiffHDRConfig(
        image_size=args.size,
        num_frames=args.frames,
        use_mask=not args.no_mask,
        use_cfa=not args.no_cfa,
    )
    model = DiffHDR(cfg).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, float]] = []

    for step in range(args.steps):
        ldr, hdr = synthesize_pair(args.frames, args.size, seed=step)
        ldr, hdr = ldr.to(device), hdr.to(device)
        opt.zero_grad(set_to_none=True)
        loss, stats = train_step(model, ldr=ldr, hdr=hdr)
        loss.backward()
        opt.step()
        stats["step"] = float(step)
        log.append(stats)
        if step % max(1, args.steps // 10) == 0:
            print(f"step {step}/{args.steps} loss={stats['loss_flow']:.4f}")

    save_checkpoint(model, out / "diffhdr.pt")
    print(json.dumps({"steps": args.steps, "log_tail": log[-3:]}, indent=2))
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = DiffHDR(DiffHDRConfig(num_frames=args.frames, image_size=args.size)).to(device).eval()
    ldr, _ = synthesize_pair(args.frames, args.size, seed=args.seed)
    ldr = ldr.to(device)
    with torch.no_grad():
        out = model(ldr)
    print(
        json.dumps(
            {
                "ldr_shape": list(ldr.shape),
                "hdr_shape": list(out.hdr.shape),
                "hdr_max": float(out.hdr.max()),
                "paper_si_hdr_pu21_piqe": 19.37,
            },
            indent=2,
        )
    )
    return 0


def _cmd_log_gamma(args: argparse.Namespace) -> int:
    hdr = __import__("ltx_trainer.diffhdr.synthetic", fromlist=["synthesize_hdr_clip"]).synthesize_hdr_clip(
        1, args.size, seed=0
    )[0]
    rows = {}
    for name in ("linear", "log", "log_gamma"):
        mapping = "log_gamma" if name == "log_gamma" else name
        err = float(vae_roundtrip_error(hdr, mapping=mapping))
        rows[name] = {"mean_abs_error": err}
    rows["paper_ours_psnr"] = 32.86
    print(json.dumps(rows, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=PAPER_TITLE)
    sub = p.add_subparsers(dest="command", required=True)

    for name, func in [
        ("knowledge", _cmd_knowledge),
        ("tables", _cmd_tables),
        ("dataset", _cmd_dataset),
    ]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=func)

    sp = sub.add_parser("train")
    sp.add_argument("-o", "--output-dir", default="diffhdr_train")
    sp.add_argument("--steps", type=int, default=100)
    sp.add_argument("--frames", type=int, default=8)
    sp.add_argument("--size", type=int, default=64)
    sp.add_argument("--lr", type=float, default=1e-4)
    sp.add_argument("--no-mask", action="store_true")
    sp.add_argument("--no-cfa", action="store_true")
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=_cmd_train)

    sp = sub.add_parser("infer")
    sp.add_argument("--frames", type=int, default=8)
    sp.add_argument("--size", type=int, default=64)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=_cmd_infer)

    sp = sub.add_parser("log-gamma")
    sp.add_argument("--size", type=int, default=64)
    sp.set_defaults(func=_cmd_log_gamma)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

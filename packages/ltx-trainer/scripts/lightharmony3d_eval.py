#!/usr/bin/env python3
"""LightHarmony3D mesh insertion CLI (Huang et al. arXiv:2603.29209)."""

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

from ltx_trainer.lightharmony3d.config import PAPER_TITLE, PAPER_URL, LightHarmony3DConfig  # noqa: E402
from ltx_trainer.lightharmony3d.dataset import dataset_summary  # noqa: E402
from ltx_trainer.lightharmony3d.model import LightHarmony3D  # noqa: E402
from ltx_trainer.lightharmony3d.pipeline import paper_report, save_checkpoint, train_step  # noqa: E402
from ltx_trainer.lightharmony3d.synthetic import synthesize_insertion_pair, synthesize_object, synthesize_scene  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    ds = dataset_summary()
    print(
        json.dumps(
            {
                "name": "LightHarmony3D",
                "paper": PAPER_TITLE,
                "url": PAPER_URL,
                "modules": ["GenEnvLighting", "HDR fusion", "Ray-decoupled shader", "Shadow ratio PBR"],
                "benchmark": ds["benchmark"],
                "baselines": ds["baselines"],
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
    cfg = LightHarmony3DConfig(image_size=args.size)
    model = LightHarmony3D(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    for step in range(args.steps):
        scene, obj, mask, gt = synthesize_insertion_pair(args.size, seed=step)
        scene, obj, mask, gt = scene.to(device), obj.to(device), mask.to(device), gt.to(device)
        opt.zero_grad(set_to_none=True)
        loss, stats = train_step(
            model,
            background=scene.unsqueeze(0),
            object_rgb=obj.unsqueeze(0),
            mask=mask.unsqueeze(0),
            target=gt.unsqueeze(0),
        )
        loss.backward()
        opt.step()
        if step % max(1, args.steps // 5) == 0:
            print(json.dumps({"step": step, **stats}))

    save_checkpoint(model, out / "lightharmony3d.pt")
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = LightHarmony3D(LightHarmony3DConfig(image_size=args.size)).to(device).eval()
    scene = synthesize_scene(args.size, seed=args.seed).to(device)
    obj, mask = synthesize_object(args.size, seed=args.seed + 1)
    obj, mask = obj.to(device), mask.to(device)
    with torch.no_grad():
        out = model(scene.unsqueeze(0), obj.unsqueeze(0), mask.unsqueeze(0))
    print(
        json.dumps(
            {
                "composite_shape": list(out.composite.shape),
                "hdr_mean": float(out.hdr_env.mean()),
                "shadow_mean": float(out.shadow_hat.mean()),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="LightHarmony3D evaluation CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("dataset")
    tr = sub.add_parser("train")
    tr.add_argument("-o", "--output-dir", default="lightharmony3d_train")
    tr.add_argument("--steps", type=int, default=100)
    tr.add_argument("--size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    inf = sub.add_parser("infer")
    inf.add_argument("--size", type=int, default=64)
    inf.add_argument("--seed", type=int, default=0)
    inf.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.cmd == "knowledge":
        return _cmd_knowledge(args)
    if args.cmd == "tables":
        return _cmd_tables(args)
    if args.cmd == "dataset":
        return _cmd_dataset(args)
    if args.cmd == "train":
        return _cmd_train(args)
    if args.cmd == "infer":
        return _cmd_infer(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

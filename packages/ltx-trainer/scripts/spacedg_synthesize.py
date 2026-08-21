#!/usr/bin/env python3
"""Synthesize SpaceDG benchmark samples (Zhou et al. arXiv:2605.22536)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.spacedg.pipeline import synthesize_benchmark_item  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="SpaceDG degradation VQA synthesis")
    p.add_argument("-o", "--output-dir", default="spacedg_out")
    p.add_argument("--scene-id", default="demo_001")
    args = p.parse_args()
    items = synthesize_benchmark_item(scene_id=args.scene_id, out_dir=args.output_dir)
    manifest = [
        {
            "question": it.qa.question,
            "answer": it.qa.answer,
            "degradation": it.degradation.value,
            "images": it.image_paths,
            "question_type": it.qa.question_type.value,
        }
        for it in items
    ]
    out = Path(args.output_dir).expanduser().resolve()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(items)} VQA instances to {out}")


if __name__ == "__main__":
    main()

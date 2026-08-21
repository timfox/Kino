"""LTX dataset prep helpers for VCap-style dense captions."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from ltx_trainer.vcap.prompts import DENSE_CAPTION_INSTRUCTION


def vcap_caption_enabled() -> bool:
    return os.environ.get("GOPEX_VCAP_CAPTION", "0").strip().lower() in ("1", "true", "yes")


def vcap_epoch() -> str:
    return os.environ.get("GOPEX_VCAP_EPOCH", "e1").strip() or "e1"


def dense_caption_instruction(*, video: bool = False) -> str:
    """Instruction string for reference-pool generation (weak-to-strong witness)."""
    if video:
        return (
            DENSE_CAPTION_INSTRUCTION
            + "\nFor video: describe temporal events, camera motion, and per-segment salient changes."
        )
    return DENSE_CAPTION_INSTRUCTION


def merge_preprocess_extra(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge caller ``extra`` with VCap block when ``GOPEX_VCAP_CAPTION=1``."""
    out = dict(extra or {})
    if vcap_caption_enabled():
        out.update(vcap_preprocess_extra(epoch=vcap_epoch()))
    return out


def summarize_qa_reports(
    work_root: str | Path,
    projects: list[str],
    *,
    datasets_root: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate ``vcap_qa.jsonl`` under each project's manifest (or work logs)."""
    work = Path(work_root).expanduser().resolve()
    ds = Path(datasets_root).expanduser().resolve() if datasets_root else None
    per_project: dict[str, Any] = {}
    all_rewards: list[float] = []
    for name in projects:
        name = name.strip()
        if not name:
            continue
        report = None
        if ds is not None:
            for man in (ds / name / "ltx_manifest", ds / name):
                cand = man / "vcap_qa.jsonl"
                if cand.is_file():
                    report = cand
                    break
        if report is None:
            cand = work / "datasets" / name / "ltx_manifest" / "vcap_qa.jsonl"
            if cand.is_file():
                report = cand
        if report is None or not report.is_file():
            per_project[name] = {"ok": False, "reason": "no vcap_qa.jsonl"}
            continue
        rewards: list[float] = []
        for line in report.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rewards.append(float(row.get("reward", 0.0)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        per_project[name] = {
            "ok": True,
            "scored": len(rewards),
            "mean_reward": round(mean_r, 4),
            "report": str(report),
        }
        all_rewards.extend(rewards)
    return {
        "projects": per_project,
        "total_scored": len(all_rewards),
        "mean_reward": round(sum(all_rewards) / len(all_rewards), 4) if all_rewards else 0.0,
    }


def prune_manifest_low_reward(
    manifest_dir: str | Path,
    *,
    min_reward: float,
    qa_report: str | Path | None = None,
) -> dict[str, Any]:
    """Drop ``dataset.json`` rows whose VCap mock reward is below ``min_reward``."""
    root = Path(manifest_dir).expanduser().resolve()
    dataset_path = root / "dataset.json"
    report_path = Path(qa_report) if qa_report else root / "vcap_qa.jsonl"
    if not dataset_path.is_file():
        return {"ok": False, "error": f"missing {dataset_path}"}
    if not report_path.is_file():
        return {"ok": False, "error": f"missing {report_path}"}

    reward_by_media: dict[str, float] = {}
    for line in report_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            mp = str(row.get("media_path") or "")
            if mp:
                reward_by_media[mp] = float(row.get("reward", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return {"ok": False, "error": "dataset.json must be a list"}

    backup = root / "dataset.pre-vcap-prune.json"
    if not backup.is_file():
        shutil.copy2(dataset_path, backup)

    kept: list[dict[str, Any]] = []
    dropped = 0
    for item in rows:
        if not isinstance(item, dict):
            continue
        mp = str(item.get("media_path") or "")
        r = reward_by_media.get(mp)
        if r is not None and r < min_reward:
            dropped += 1
            continue
        kept.append(item)

    dataset_path.write_text(json.dumps(kept, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "kept": len(kept),
        "dropped": dropped,
        "min_reward": min_reward,
        "backup": str(backup),
    }


def vcap_preprocess_extra(
    *,
    epoch: str = "e1",
    reward_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Extra fields merged into ``preprocess_meta.json`` via ``write_preprocess_meta(..., extra=...)``."""
    weights = reward_weights or {"wcorr": 0.05, "wcomp": 0.04, "wtxt": 0.01}
    return {
        "vcap": {
            "enabled": True,
            "training_epoch": epoch,
            "caption_style": "dense_witness_adjudicator",
            "reward_weights": weights,
            "paper": "arXiv:2605.28023",
        }
    }


def score_caption_pair(
    reference: str,
    policy: str,
    *,
    cfg: Any = None,
) -> dict[str, Any]:
    """Score (reference, policy) with mock witness-adjudicator (for dataset QA / smoke)."""
    from ltx_trainer.vcap.config import VCapConfig
    from ltx_trainer.vcap.hypergeometric import fact_counts_from_scores
    from ltx_trainer.vcap.rewards import estimate_fact_counts_from_captions, mock_judge_scores, sentence_reward

    cfg = cfg or VCapConfig()
    scores = mock_judge_scores(policy, reference, cfg=cfg)
    r = sentence_reward(scores, cfg)
    c, n, m = estimate_fact_counts_from_captions(reference, policy, N=cfg.latent_facts_N)
    hg = fact_counts_from_scores(c=c, n_total=n, m_witness=m, N=cfg.latent_facts_N)
    return {
        "reward": r,
        "scores": {"scorr": scores.scorr, "scomp": scores.scomp, "stxt": scores.stxt},
        "hypergeometric": hg,
        "analysis": scores.analysis,
    }


def attach_vcap_caption_meta(meta: dict[str, Any], *, video: bool = True) -> dict[str, Any]:
    """Add VCap dense-caption hints to clip meta when ``GOPEX_VCAP_CAPTION=1``."""
    if not vcap_caption_enabled():
        return meta
    out = dict(meta)
    out["vcap"] = {
        "epoch": vcap_epoch(),
        "dense_instruction": dense_caption_instruction(video=video),
    }
    return out


def vcap_user_prompt_lines(meta: dict[str, Any] | None) -> list[str]:
    """Extra user-message lines for Gemma vision captioning (witness-adjudicator style)."""
    if not vcap_caption_enabled():
        return []
    block = (meta or {}).get("vcap")
    if isinstance(block, dict) and block.get("dense_instruction"):
        instr = str(block["dense_instruction"]).strip()
    else:
        instr = dense_caption_instruction(video=True)
    return [
        "--- VCap dense captioning (fact-level; arXiv:2605.28023) ---",
        instr,
        "Minimize omissions and hallucinations: every object, attribute, relation, and motion "
        "you state should be visible in the frames. No self-evaluation or meta text.",
        "--- End VCap ---",
    ]


def qa_manifest_directory(
    manifest_dir: str | Path,
    *,
    reference_dataset: str | Path | None = None,
    min_reward: float = 0.0,
    report_path: str | Path | None = None,
    cfg: Any = None,
) -> dict[str, Any]:
    """Score each row in ``dataset.json`` against a reference caption pool (witness QA)."""
    from ltx_trainer.vcap.config import VCapConfig

    cfg = cfg or VCapConfig()
    root = Path(manifest_dir).expanduser().resolve()
    dataset_path = root / "dataset.json"
    if not dataset_path.is_file():
        return {"ok": False, "error": f"missing {dataset_path}", "scored": 0}

    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return {"ok": False, "error": "dataset.json must be a list", "scored": 0}

    ref_by_media: dict[str, str] = {}
    if reference_dataset is not None:
        ref_path = Path(reference_dataset).expanduser().resolve()
        if ref_path.is_file():
            ref_rows = json.loads(ref_path.read_text(encoding="utf-8"))
            if isinstance(ref_rows, list):
                for item in ref_rows:
                    if isinstance(item, dict):
                        mp = str(item.get("media_path") or "")
                        cap = str(item.get("caption") or "").strip()
                        if mp and cap:
                            ref_by_media[mp] = cap

    report_lines: list[dict[str, Any]] = []
    rewards: list[float] = []
    below = 0
    for item in rows:
        if not isinstance(item, dict):
            continue
        media = str(item.get("media_path") or "")
        policy = str(item.get("caption") or "").strip()
        if not media or not policy:
            continue
        reference = ref_by_media.get(media) or policy
        scored = score_caption_pair(reference, policy, cfg=cfg)
        reward = float(scored["reward"])
        rewards.append(reward)
        if reward < min_reward:
            below += 1
        line = {"media_path": media, "reward": reward, **scored}
        report_lines.append(line)

    out_path = Path(report_path) if report_path else root / "vcap_qa.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    mean_r = sum(rewards) / len(rewards) if rewards else 0.0
    return {
        "ok": True,
        "manifest_dir": str(root),
        "scored": len(rewards),
        "mean_reward": round(mean_r, 4),
        "below_min_reward": below,
        "min_reward": min_reward,
        "report": str(out_path),
        "reference_rows": len(ref_by_media),
    }


def tag_precomputed_projects(
    work_root: str | Path,
    projects: list[str],
    *,
    epoch: str | None = None,
) -> list[str]:
    """Write VCap block into each project's ``preprocess_meta.json`` under work_root/precomputed."""
    epoch = epoch or vcap_epoch()
    touched: list[str] = []
    base = Path(work_root).expanduser().resolve() / "precomputed"
    for name in projects:
        name = name.strip()
        if not name:
            continue
        pre = base / name
        if not pre.is_dir():
            continue
        attach_vcap_to_preprocess_meta(pre, epoch=epoch)
        touched.append(name)
    return touched


def attach_vcap_to_preprocess_meta(
    precomputed_root: str | Path,
    *,
    epoch: str = "e1",
) -> Path:
    """Merge VCap metadata into an existing preprocess tree."""
    from ltx_trainer.preprocess_meta import read_preprocess_meta, write_preprocess_meta

    root = Path(precomputed_root).expanduser().resolve()
    meta = read_preprocess_meta(root) or {}
    model_path = meta.get("model_path", "")
    text_encoder = meta.get("text_encoder_path", "")
    extra = dict(meta.get("extra") or {})
    extra.update(vcap_preprocess_extra(epoch=epoch))
    return write_preprocess_meta(
        root,
        model_path=model_path or str(root),
        text_encoder_path=text_encoder or str(root),
        flat_dim_bridge_rank=meta.get("flat_dim_bridge_rank"),
        dataset_file=meta.get("dataset_file"),
        extra=extra,
    )

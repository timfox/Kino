"""Load LiveBrowseComp from Hub cache or local JSONL."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

from ltx_trainer.livebrowsecomp.config import LiveBrowseCompConfig
from ltx_trainer.livebrowsecomp.corpus import LiveBrowseCompItem, item_from_row
from ltx_trainer.livebrowsecomp.crypto import decrypt_string

_JSONL_NAME = "LiveBrowseComp.jsonl"


def default_cache_dir() -> Path:
    raw = os.environ.get("GOPEX_LIVEBROWSECOMP_CACHE", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".cache" / "gopex" / "livebrowsecomp").resolve()


def resolve_jsonl_path(
    path: str | Path | None = None,
    *,
    download: bool = True,
    cfg: LiveBrowseCompConfig | None = None,
) -> Path:
    """Resolve JSONL: explicit path, cache copy, or Hub download."""
    if path is not None:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"LiveBrowseComp JSONL not found: {p}")
        return p

    cfg = cfg or LiveBrowseCompConfig()
    cached = default_cache_dir() / _JSONL_NAME
    if cached.is_file():
        return cached

    if not download:
        raise FileNotFoundError(
            f"No {_JSONL_NAME} at {cached}. Set GOPEX_LIVEBROWSECOMP_CACHE or pass --path."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("huggingface_hub required to download LiveBrowseComp") from e

    hub_path = hf_hub_download(
        cfg.hub_dataset,
        _JSONL_NAME,
        repo_type="dataset",
    )
    src = Path(hub_path)
    cached.parent.mkdir(parents=True, exist_ok=True)
    if not cached.exists() or cached.stat().st_mtime < src.stat().st_mtime:
        cached.write_bytes(src.read_bytes())
    return cached


def iter_rows(
    path: str | Path,
    *,
    decrypt: bool = True,
) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if decrypt:
                row = {
                    **row,
                    "problem": decrypt_string(row["problem"]),
                    "answer": decrypt_string(row["answer"]),
                }
            yield row


def load_items(
    path: str | Path | None = None,
    *,
    decrypt: bool = True,
    download: bool = True,
    limit: int | None = None,
) -> list[LiveBrowseCompItem]:
    jsonl = resolve_jsonl_path(path, download=download)
    out: list[LiveBrowseCompItem] = []
    for row in iter_rows(jsonl, decrypt=decrypt):
        out.append(item_from_row(row))
        if limit is not None and len(out) >= limit:
            break
    return out


def dataset_stats(items: list[LiveBrowseCompItem] | None = None, **load_kw: Any) -> dict[str, Any]:
    if items is None:
        items = load_items(**load_kw)
    lengths_q = [len(i.problem) for i in items]
    lengths_a = [len(i.answer) for i in items]
    return {
        "count": len(items),
        "idx_min": min(i.idx for i in items) if items else None,
        "idx_max": max(i.idx for i in items) if items else None,
        "problem_chars_mean": round(sum(lengths_q) / len(lengths_q), 1) if lengths_q else 0,
        "answer_chars_mean": round(sum(lengths_a) / len(lengths_a), 1) if lengths_a else 0,
    }


def checkout_status() -> dict[str, Any]:
    cache = default_cache_dir()
    jsonl = cache / _JSONL_NAME
    return {
        "env_var": "GOPEX_LIVEBROWSECOMP_CACHE",
        "cache_dir": str(cache),
        "jsonl_present": jsonl.is_file(),
        "jsonl_path": str(jsonl) if jsonl.is_file() else None,
        "hub_dataset": LiveBrowseCompConfig().hub_dataset,
    }

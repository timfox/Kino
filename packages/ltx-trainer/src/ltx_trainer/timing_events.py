"""Low-overhead JSONL timing events for CID scheduling and ETA estimates."""
from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import Any

def record(output_dir: str | Path, event: str, *, duration_s: float | None = None, **fields: Any) -> None:
    try:
        root=Path(output_dir); root.mkdir(parents=True, exist_ok=True)
        row={"schema":"gopex.timing_event/v1","event":event,"ts":time.time(),"pid":os.getpid()}
        if duration_s is not None: row["duration_s"]=round(float(duration_s),4)
        row.update(fields)
        with (root/"timing_events.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    except OSError:
        pass

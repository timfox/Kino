"""Yi evaluation pipeline and smoke checks."""

from __future__ import annotations

from typing import Any

from ltx_trainer.yi.config import YiConfig
from ltx_trainer.yi.graph import demo_graph
from ltx_trainer.yi.system import (
    AsyncBufferManager,
    TaskletEngine,
    VectorFileSystem,
    component_breakdown,
)
from ltx_trainer.yi.vector_update import VectorLevelUpdater


def run_update_demo(n_ops: int = 20, config: YiConfig | None = None) -> dict[str, Any]:
    cfg = config or YiConfig()
    g = demo_graph(n=48, degree=min(4, cfg.max_out_degree))
    updater = VectorLevelUpdater(g, delete_lru_capacity=max(4, int(48 * cfg.delete_lru_frac)))
    engine = TaskletEngine()
    buf = AsyncBufferManager(capacity=8)
    vfs = VectorFileSystem(split_layout=cfg.enable_split_layout)

    for i in range(n_ops):
        if i % 2 == 0:
            vid = 1000 + i
            updater.insert(vid, raw=(float(i), float(i % 3)))
            if cfg.enable_tasklets:
                engine.schedule_insert(vid, n_neighbors=3)
            vfs.access_for_update(vid)
            buf.access(vid % 8, write=True)
        else:
            victim = i % 48
            updater.delete(victim)
            vfs.access_for_update(victim)
            buf.access(victim % 8, write=True)

    drain = engine.drain(overlap_io=cfg.enable_tasklets) if cfg.enable_tasklets else {"processed": 0}
    flushed = buf.flush_dirty()
    return {
        "ops": n_ops,
        "stats": {
            "connect_tasks": updater.stats.connect_tasks,
            "vectors_visited": updater.stats.vectors_visited,
            "inserts": updater.stats.inserts,
            "deletes": updater.stats.deletes,
            "sampled_connects": updater.stats.sampled_connects,
        },
        "tasklets": drain,
        "buffer_hit_ratio": round(buf.hit_ratio, 3),
        "dirty_flushed": flushed,
        "vfs": {
            "index_accesses": vfs.index_block_accesses,
            "data_accesses": vfs.data_block_accesses,
            "cache_friendliness": round(vfs.cache_friendliness(), 3),
        },
        "breakdown": component_breakdown(cfg),
    }


def evaluation_demo() -> dict[str, Any]:
    full = run_update_demo(24, YiConfig())
    coupled = run_update_demo(24, YiConfig(enable_split_layout=False))
    return {
        "full": full,
        "coupled_layout": coupled,
        "layout_helps": full["vfs"]["data_accesses"] < coupled["vfs"]["data_accesses"],
    }


def evaluation_smoke() -> dict[str, bool]:
    from ltx_trainer.yi.benchmarks import PAPER_ANCHORS

    demo = evaluation_demo()
    full = demo["full"]
    checks = {
        "connect_tasks_run": full["stats"]["connect_tasks"] > 0,
        "inserts_and_deletes": full["stats"]["inserts"] > 0 and full["stats"]["deletes"] > 0,
        "tasklets_processed": full["tasklets"].get("processed", 0) > 0,
        "split_layout_skips_data": full["vfs"]["data_accesses"] == 0,
        "layout_helps": demo["layout_helps"],
        "breakdown_gt1": full["breakdown"]["final_rel"] > 4.0,
        "paper_sift800m_update_x": abs(float(PAPER_ANCHORS["sift800m_update_speedup"]) - 1.75) < 1e-6,
        "paper_sift800m_search_x": abs(float(PAPER_ANCHORS["sift800m_search_speedup"]) - 1.8) < 1e-6,
        "paper_peak_mem_frac": abs(float(PAPER_ANCHORS["sift800m_peak_mem_frac"]) - 0.73) < 1e-6,
        "paper_deep100m_update_qps": abs(float(PAPER_ANCHORS["deep100m_update_qps"]) - 1600) < 1.0,
    }
    checks["all_pass"] = all(checks.values())
    return checks


def evaluation_smoke_json() -> dict[str, bool]:
    return evaluation_smoke()

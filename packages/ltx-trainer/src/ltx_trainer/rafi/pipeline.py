"""RaFI framework card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.rafi.config import RafiConfig
from ltx_trainer.rafi.device import pack_sort_keys, sort_rays_by_destination
from ltx_trainer.rafi.forward import forward_rays_smoke, tally_send_segments
from ltx_trainer.rafi.host import HostContext
from ltx_trainer.rafi.layout import LIMITATIONS
from ltx_trainer.rafi.paper_tables import fig8_bandwidth_utilization, ray_throughput
from ltx_trainer.rafi.types import NBodyParticle, SchlierenFwdRay, SimpleRay, StreamlineParticle


def framework_card(cfg: RafiConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RafiConfig()
    return {
        "name": "RaFI",
        "paper": cfg.paper_arxiv,
        "license": cfg.license,
        "idea": (
            "CUDA + MPI work-forwarding: kernels call emitOutgoing(item, destRank); "
            "host forwardRays() sorts by destination, Alltoallv exchanges GPU queues, "
            "and reduce reports distributed termination."
        ),
        "device_api": ["numIncoming", "getIncoming", "emitOutgoing"],
        "host_api": ["resizeRayQueues", "getDeviceInterface", "forwardRays"],
        "example_apps": list(cfg.example_apps),
        "related": list(cfg.related),
        "local_gpus": list(cfg.local_gpus),
        "limitations": list(LIMITATIONS),
    }


def dual_gpu_rank_map() -> dict[str, Any]:
    """Suggested 2-rank layout for RTX Pro 6000 + Pro 4000 (one GPU per rank)."""
    return {
        "rank0": {"device": "cuda:0", "label": "NVIDIA RTX Pro 6000"},
        "rank1": {"device": "cuda:1", "label": "NVIDIA RTX Pro 4000"},
        "note": "Requires CUDA-aware MPI and one process per GPU; stub does not launch MPI.",
    }


def benchmarks_bundle(cfg: RafiConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RafiConfig()
    return {
        "fig8_bandwidth": fig8_bandwidth_utilization(),
        "ray_throughput": ray_throughput(),
        "intranode_sustained_gbps": cfg.intranode_sustained_gbps,
        "internode_sustained_gbps": cfg.internode_sustained_gbps,
    }


def evaluation_demo(cfg: RafiConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RafiConfig()
    num_ranks = 2

    # Device emit smoke
    ctx0 = HostContext[SimpleRay](rank=0, num_ranks=num_ranks, max_capacity=16)
    ctx1 = HostContext[SimpleRay](rank=1, num_ranks=num_ranks, max_capacity=16)
    dev0 = ctx0.get_device_interface()
    dev0._queues.incoming = [
        SimpleRay((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0),
        SimpleRay((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1),
    ]
    dev0.emit_outgoing(SimpleRay((2.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 2), dest=1)
    dev0.emit_outgoing(SimpleRay((0.5, 0.5, 0.0), (0.0, 0.0, 1.0), 3), dest=0)

    dev1 = ctx1.get_device_interface()
    dev1._queues.incoming = [SimpleRay((3.0, 0.0, 0.0), (0.0, -1.0, 0.0), 4)]
    dev1.emit_outgoing(SimpleRay((3.0, 1.0, 0.0), (0.0, 0.0, -1.0), 5), dest=0)

    per_out = [list(ctx0._queues.outgoing), list(ctx1._queues.outgoing)]
    per_dest = [list(ctx0._queues.destinations), list(ctx1._queues.destinations)]

    total = ctx0.forward_rays(per_rank_outgoing=per_out, per_rank_dests=per_dest)
    received_rank0 = len(ctx0._queues.incoming)

    # Sort keys
    keys = pack_sort_keys([1, 0, 1, 0], [0, 1, 2, 3])
    sorted_rays, sorted_dests = sort_rays_by_destination(
        [SimpleRay((0, 0, 0), (1, 0, 0), i) for i in range(4)],
        [1, 0, 1, 0],
    )
    send_count, send_offset = tally_send_segments(sorted_dests, 2)

    # Schlieren / streamline / n-body type sizes
    sch = SchlierenFwdRay((0, 0, 0), (0, 0, 1), 0.0, 0, 0.0, (0.0, 0.0, 0.0))
    part = StreamlineParticle(7, (1.0, 2.0, 3.0))
    nb = NBodyParticle(1.0, (0, 0, 0), (0, 1, 0))

    bandwidth = fig8_bandwidth_utilization()
    throughput = ray_throughput()

    return {
        "num_incoming_rank0": dev0.num_incoming(),
        "emitted_ok": dev0.emit_outgoing(SimpleRay((0, 0, 0), (1, 0, 0), 99), dest=0),
        "forward_total_rays": total,
        "received_rank0": received_rank0,
        "sort_keys_len": int(keys.numel()),
        "sorted_dest_order": sorted_dests,
        "send_count": send_count,
        "send_offset": send_offset,
        "schlieren_pixel": sch.pixel_id,
        "streamline_id": part.particle_id,
        "nbody_mass": nb.mass,
        "bandwidth": bandwidth,
        "throughput": throughput,
        "paper_intranode_gbps": cfg.intranode_sustained_gbps,
        "dual_gpu": dual_gpu_rank_map(),
    }

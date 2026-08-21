"""CPU smoke for OMAF fast multirate encoding stub."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.oma_fme.benchmarks import benchmarks_bundle
from ltx_trainer.oma_fme.datasets import sjtu_dataset_card
from ltx_trainer.oma_fme.metrics import bdet_stub, delta_t_serial
from ltx_trainer.oma_fme.pipeline import run_pipeline
from ltx_trainer.oma_fme.projections import erp_to_sphere_direction, erp_uv_grid, stub_erp_to_cmp_faces
from ltx_trainer.oma_fme.synthetic import synthetic_variant_comparison
from ltx_trainer.oma_fme.config import OmaFmeConfig


def evaluation_smoke() -> dict[str, Any]:
    erp = torch.rand(1, 3, 256, 512)
    u, v = erp_uv_grid(256, 512)
    dirs = erp_to_sphere_direction(u, v)
    faces = stub_erp_to_cmp_faces(erp, face_size=64)

    cmp_out = run_pipeline(OmaFmeConfig(variant="CMP-PRA"), erp_batch=erp)
    erp_out = run_pipeline(OmaFmeConfig(variant="ERP-CRC"), erp_batch=erp)

    synth = synthetic_variant_comparison()
    bundle = benchmarks_bundle()

    t_m = torch.tensor([40.0, 80.0, 120.0])
    t_r = torch.tensor([100.0, 200.0, 300.0])
    bdet = bdet_stub(t_m, t_r)

    return {
        "status": "ok",
        "directions_shape": list(dirs.shape),
        "cmp_face_count": len(faces),
        "cmp_pipeline_encodes": cmp_out["total_encodes"],
        "erp_pipeline_encodes": erp_out["total_encodes"],
        "synthetic_delta_T_S_erp_crc": synth["delta_T_S"]["ERP-CRC"],
        "table3_bd_wspsnr_cmp_pra": bundle["table3_cmp_pra_hq_avg"]["BD_WSPSNR_db"],
        "bdet_proxy": bdet,
        "sjtu_sequences": sjtu_dataset_card()["sequence_count"],
        "delta_t_serial": delta_t_serial(
            [synth["CMP-PRA-HQ"]["serial_s"]],
            [synth["ERP-Default"]["serial_s"]],
        ),
    }

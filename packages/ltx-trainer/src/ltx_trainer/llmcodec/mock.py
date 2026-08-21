"""LLMCodec CPU reference smokes (arXiv:2606.05861)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.llmcodec.affine import learn_affine_stub
from ltx_trainer.llmcodec.codec import compress_yuv420_stub, rd_curve_stub
from ltx_trainer.llmcodec.compress import compress_weight_matrix
from ltx_trainer.llmcodec.config import CodecType, CodingProfile
from ltx_trainer.llmcodec.mapping import weight_to_yuv420


def run_affine_smoke(*, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1.0, size=(64, 64))
    w[0, 0] = 12.0  # outlier
    t, err = learn_affine_stub(w, seed=seed)
    return {"affine_error": round(err, 5), "t_shape": list(t.shape), "had_outlier": abs(float(w[0, 0])) > 5}


def run_compress_layer_smoke(*, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.02, size=(128, 128))
    w[0, 0] = 0.5  # mild outlier
    with_affine = compress_weight_matrix(w, qp=10, use_affine=True, seed=seed)
    without = compress_weight_matrix(w, qp=10, use_affine=False, seed=seed)
    return {
        "with_affine_mse": round(with_affine["recon_mse"], 6),
        "without_affine_mse": round(without["recon_mse"], 6),
        "avg_bitwidth": with_affine["avg_bitwidth"],
        "psnr_y": with_affine["psnr_y"],
        "affine_error": with_affine["affine_error"],
    }


def run_codec_ablation_smoke(*, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.03, size=(128, 128))
    yuv, _ = weight_to_yuv420(w)
    ai = compress_yuv420_stub(yuv, qp=12, profile=CodingProfile.ALL_INTRA, codec=CodecType.VVC)
    ld = compress_yuv420_stub(yuv, qp=12, profile=CodingProfile.LOW_DELAY, codec=CodecType.VVC)
    vvc = compress_yuv420_stub(yuv, qp=12, codec=CodecType.VVC)
    jpeg = compress_yuv420_stub(yuv, qp=12, codec=CodecType.JPEG)
    return {
        "all_intra_psnr": ai.psnr_y,
        "low_delay_psnr": ld.psnr_y,
        "vvc_psnr": vvc.psnr_y,
        "jpeg_psnr": jpeg.psnr_y,
        "vvc_beats_jpeg": vvc.psnr_y > jpeg.psnr_y,
        "ai_beats_ld": ai.psnr_y >= ld.psnr_y,
    }


def run_rd_curve_smoke(*, seed: int = 0) -> dict[str, Any]:
    curve = rd_curve_stub(seed=seed)
    low_qp = curve[0]
    high_qp = curve[-1]
    return {
        "n_points": len(curve),
        "low_qp_psnr": low_qp["psnr_y"],
        "high_qp_psnr": high_qp["psnr_y"],
        "monotonic_bitwidth": low_qp["avg_bitwidth"] > high_qp["avg_bitwidth"],
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    affine = run_affine_smoke(seed=seed)
    layer = run_compress_layer_smoke(seed=seed + 1)
    ablation = run_codec_ablation_smoke(seed=seed + 2)
    rd = run_rd_curve_smoke(seed=seed + 3)
    from ltx_trainer.llmcodec.pipeline import fig1_llama3_2bit_anchors

    anchors = fig1_llama3_2bit_anchors()
    paper_ok = (
        anchors["LLMCodec_ppl_wikitext2"] < anchors["FlatQuant_ppl_wikitext2"]
        and anchors["LLMCodec_avg_accuracy"] > anchors["FlatQuant_avg_accuracy"]
    )
    return {
        "affine": affine,
        "layer_compress": layer,
        "codec_ablation": ablation,
        "rd_curve": rd,
        "paper_2bit_llama3_ok": paper_ok,
        "pipeline_ok": (
            ablation["vvc_beats_jpeg"]
            and ablation["ai_beats_ld"]
            and rd["monotonic_bitwidth"]
            and layer["affine_error"] < 1.0
        ),
    }

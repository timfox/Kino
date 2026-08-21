"""DiT renderer incremental CFG-style guidance — Sec. 4.2, Eq. (8)–(12)."""

from __future__ import annotations


def incremental_guidance_prediction(
    eps_none: float,
    eps_vid: float,
    eps_vid_img: float,
    eps_txt_vid_img: float,
    eps_full: float,
    *,
    omega_vid: float,
    omega_img: float,
    omega_txt: float,
    omega_tgt: float,
) -> float:
    r"""ˆε = ε_∅ + ω_vid Δ_vid + ... (Eq. 12)."""
    d_vid = eps_vid - eps_none
    d_img = eps_vid_img - eps_vid
    d_txt = eps_txt_vid_img - eps_vid_img
    d_tgt = eps_full - eps_txt_vid_img
    return eps_none + omega_vid * d_vid + omega_img * d_img + omega_txt * d_txt + omega_tgt * d_tgt


def guidance_scales_t2v() -> dict[str, float | None]:
    """Table 5 — T2V row."""
    return {"omega_txt": 4.0, "omega_vid": None, "omega_img": 1.0, "omega_tgt": 1.0}


def guidance_scales_v2v() -> dict[str, float | None]:
    """Table 5 — V2V row."""
    return {"omega_txt": 4.0, "omega_vid": 1.25, "omega_img": 1.25, "omega_tgt": 0.5}

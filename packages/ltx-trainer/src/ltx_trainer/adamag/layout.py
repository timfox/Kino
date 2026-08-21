"""AdaMaG concepts and limitations."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "Divergence term ∇·g is not computed at inference (memory); schedule + β approximate control.",
    "Analysis assumes rectified-flow velocity–score relation; DDPM/ε-prediction needs score-space view.",
    "Evaluated on text-to-image (SD3, SD3.5, Flux); no video or 3D extension in this stub.",
    "No additional NFE vs CFG (unlike Rect-CFG++ predictor–corrector).",
)

CONSERVATION_EQ5: str = (
    "∇·g + g⊤∇log p = 0  →  divergence term (i) + score-parallel flux (ii)"
)

PIPELINE_STEPS: tuple[str, ...] = (
    "Sample latent x_t; get v_u, v_c from frozen flow model",
    "g = v_c - v_u (CFG residual); n_t = a_t x - v_c",
    "Decompose g → g∥ (parallel to n_t) + g⊥ (orthogonal)",
    "ω(t) = max(ω_min, ω_ref (1-t)^γ) late-time attenuation",
    "v_guided = v_u + ω(t)(g⊥ + β g∥)  — plug-and-play, same NFE as CFG",
)

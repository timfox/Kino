"""Foley-Omni scope and limitations."""

LIMITATIONS: tuple[str, ...] = (
    "No pretrained Foley-Omni DiT, Mel VAE, or BigVGAN weights in this stub.",
    "No Gemini 2.5 Pro or Bandit separation at runtime.",
    "V2ST-Bench release URLs/metadata only — no 300-clip download pipeline.",
    "Generates a single mixed latent track; no per-component balance UI.",
    "LTX-2 joint AV video+audio path is separate; use this stub for V2ST audio-side planning.",
)

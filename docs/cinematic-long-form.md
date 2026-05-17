# Cinematic long-form workflow (Kino)

Kino prioritizes **long coherent takes**, **language-driven blocking and camera**, and **edit-friendly generation**—not real-time interactive world simulation. This page ties together the knobs and pipelines that support that goal.

## 1. Long text context (Gemma)

The diffusion stack is conditioned on embeddings from the **Gemma 4** text encoder. Effective prompt length is capped by `min(HF max_position_embeddings, LTX_GEMMA_ENCODE_CAP)` with a **default cap of 8192** tokens so long shot lists and scene briefs fit in one encode.

- **Raise or lower the cap** with environment variable `LTX_GEMMA_ENCODE_CAP` (integer). Use a lower value only if you need to match legacy behavior or reduce encoder VRAM; see [Text Encoding (Gemma)](../packages/ltx-core/README.md#text-encoding-gemma).
- **Use one strong prompt** per generation: chronological, single voice, explicit camera and layout language (see below)—rather than many tiny unrelated prompts.

## 2. Prompt discipline

Think in **shot-list prose**: one flowing paragraph, literal and ordered in time.

- **Action first**, then wardrobe, environment, light, and **camera** (lens feel, height, move: dolly, pan, static, etc.) in **natural language**—not joystick or matrix inputs.
- **Spatial anchors**: where subjects sit relative to the room, background landmarks, and eyelines so the model can hold a stable “stage.”
- **LTX blog guidance** ([How to prompt for LTX-2](https://ltx.video/blog/how-to-prompt-for-ltx-2)) remains a good default; with Gemma 4 and a high encode cap you can go **longer than ~200 words** when the extra text adds real structure (beats, reversals, camera progression). Prefer density over fluff.

Optional **automatic prompt enhancement** (`enhance_prompt` in pipelines) can help polish wording; keep creative direction in your base prompt.

## 3. Two-stage and HQ samplers

For **production-quality** video, use **two-stage** pipelines: base generation plus spatial upsampling and distilled refinement.

| Goal | Pipeline | Notes |
|------|----------|--------|
| Default production quality | `TI2VidTwoStagesPipeline` | Euler-based; broadest recipe. |
| Higher quality / often fewer steps | `TI2VidTwoStagesHQPipeline` | **res_2s** second-order sampler in both stages; different guidance defaults (see pipelines README). |

Avoid **single-stage** and **pure distilled** paths when the priority is maximum fidelity over a long clip; they are better for speed and iteration.

## 4. Consistency and identity

When the same subject or hero asset must stay recognizable across motion and edits:

- **`ConsistencyPipeline`** — hero image, optional extra keyframes, optional reference video/mask; dispatches to TI2Vid or IC-LoRA as needed. Use presets (`balanced`, `strong_identity`, `masked_subject`) per the root README.

Keyframes are **narrative anchors**, not game-state: they reinforce who/what/when across the timeline.

## 5. Retake-style workflows for long takes

A long cinematic piece is usually **assembled**:

1. Generate **segments** (each with a tight prompt and optional keyframes).
2. **Review**; for weak sections only, run **`RetakePipeline`** on a **time region** of the existing clip to regenerate video (and optionally audio) while keeping the rest.
3. Optionally use **`KeyframeInterpolationPipeline`** between approved stills to bridge motion.

This matches editorial practice: fix shots, not the whole film.

## 6. Related reading

- [LTX-Pipelines README](../packages/ltx-pipelines/README.md) — pipeline matrix, CLI entrypoints, denoising tips.
- [ltx-core README (Gemma)](../packages/ltx-core/README.md) — `LTX_GEMMA_ENCODE_CAP`, checkpoint layout.

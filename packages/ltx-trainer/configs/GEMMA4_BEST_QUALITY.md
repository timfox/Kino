# Gemma 4 + LTX 2.3 — best quality path

Use **bridge rank 512** (highest practical low-rank bottleneck before the useless dense random bridge). End goal: **native** checkpoint after fold + **LoRA rank 128** with FFN adapters.

## Quality ladder (best → acceptable)

| Step | What | Config / command |
|------|------|------------------|
| 1 | Geometry probe | `python tools/ltx_gemma_flat_probe.py --ltx … --gemma …` |
| 2 | Preprocess @ rank **512** | `process_dataset.py … --flat-dim-bridge-rank 512` |
| 3 | Strict audit | `ltx_precomputed_audit.py --strict` |
| 4 | **Phase 1a** — train bridge + aggregates | `ltx2_text_stack_gemma4_rank512.yaml` |
| 5 | **Fold** → native LTX | `fold_flat_dim_bridge.py --flat-dim-bridge-rank 512` |
| 6 | Probe native (exit **0**) | same probe on folded checkpoint |
| 7 | Re-preprocess **native** | same `dataset.json`, **native** `model_path`, **no** bridge rank; `rm -rf conditions/` first |
| 8 | **Phase 2** — native LoRA 128 | `ltx2_av_lora_gemma4_native_best.yaml` |

**Interim** (if you train DiT before fold): `ltx2_av_lora_gemma4_rank512_connectors.yaml` after step 2.

**Do not use** `--allow-dense-bridge` for best quality (30GB GPU, untrainable random projection).

## Helper CLI

```bash
cd kino/packages/ltx-trainer
python scripts/gemma4_best_quality.py plan
python scripts/gemma4_best_quality.py preprocess --run          # rank 512
python scripts/gemma4_best_quality.py audit --run
python scripts/gemma4_best_quality.py phase1a --run            # free GPU first
python scripts/gemma4_best_quality.py fold --text-stack …/text_stack_weights_step_04000.safetensors --run
python scripts/gemma4_best_quality.py preprocess-native --run  # after rm conditions/
python scripts/gemma4_best_quality.py phase2-native --run
```

## Your Prelinger dataset (current state)

If `conditions/` were built with the **dense** bridge, delete them and re-embed at rank 512 before Phase 1a/0:

```bash
rm -rf /run/media/tim/Datasets/datasets/prelinger-archives-open/.precomputed/conditions
```

If `latents/` used old paths (`data/foo.pt` vs `prelinger-archives-open/data/clips/...`), remove stale latents and run preprocess with the **same** `dataset.json` as captions:

```bash
rm -rf …/.precomputed/latents …/.precomputed/audio_latents
python scripts/process_dataset.py … --flat-dim-bridge-rank 512 --skip-existing …
```

Or finish latents only while fixing captions separately:

```bash
python scripts/process_dataset.py … --latents-only --skip-existing …
```

## VRAM

| Stage | ~VRAM |
|-------|--------|
| Preprocess captions @ rank 512 | Gemma 26B bf16 + small bridge |
| Phase 1a text stack | Gemma + bridge trainable + frozen DiT |
| Phase 0/2 LoRA 128 | DiT LoRA + connectors |

Stop **vLLM** and other GPU jobs before training steps.

## Inference (after training)

```bash
python scripts/inference.py \
  --checkpoint /path/to/ltx-2.3-22b-gemma4-native.safetensors \
  --text-encoder-path /path/to/gemma-4-snapshot \
  --lora-path …/lora_weights_step_XXXXX.safetensors \
  --prompt "…" --output out.mp4
```

Use `--flat-dim-bridge-rank 512` only if still on non-native checkpoint + interim weights.

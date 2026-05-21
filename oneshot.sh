#!/usr/bin/env bash
set -euo pipefail

# Gemma profile (default: local Gemma 4 31B IT + bridge rank 512):
#   GEMMA_PROFILE=gemma31b ./kino/oneshot.sh
#   GEMMA_PROFILE=gemma26b_a4b ./kino/oneshot.sh   # prior 26B-A4B + rank 32
#
# Datasets only: PREPROCESS_ONLY=1 ./kino/oneshot.sh

export GOPEX_REPO="${GOPEX_REPO:-$HOME/GopexLLC}"
# shellcheck disable=SC1091
source "$GOPEX_REPO/scripts/gopex-gemma-env.sh"

export DISTILLED_LORA="${DISTILLED_LORA:-$HOME/ComfyUI/models/loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors}"
export UPSAMPLER="${UPSAMPLER:-$HOME/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors}"
export VLLM_PORT="${VLLM_PORT:-8001}"
# Trainer saves lora_weights_step_{step:05d}.safetensors → step 6000 is 06000, not 006000.
_resolve_lora_ckpt() {
  if [[ -n "${LORA_CKPT:-}" && -f "$LORA_CKPT" ]]; then
    echo "$LORA_CKPT"
    return
  fi
  local ckpt_dir="${PHASE0_RUN}/checkpoints"
  local latest
  latest="$(ls -t "$ckpt_dir"/lora_weights_step_*.safetensors 2>/dev/null | head -1)"
  if [[ -n "$latest" ]]; then
    echo "$latest"
    return
  fi
  echo "${ckpt_dir}/lora_weights_step_06000.safetensors"
}
export LORA_CKPT="${LORA_CKPT:-$(_resolve_lora_ckpt)}"
export INFER_OUT="${WORK_ROOT}/samples/phase0_smoke.mp4"
export HQ_OUT="${WORK_ROOT}/samples/hq_two_stage.mp4"
export PROMPT="${PROMPT:-Archival black and white film, steady camera, clear motion, documentary style.}"

cd "$GOPEX_REPO"
echo "Gemma profile: ${GEMMA_PROFILE}  encoder=${GEMMA}  bridge_rank=${BRIDGE_RANK}  work_root=${WORK_ROOT}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
TRAINER="$GOPEX_REPO/kino/packages/ltx-trainer"
export PYTHONPATH="$TRAINER/src:$GOPEX_REPO/kino/packages/ltx-pipelines/src:$GOPEX_REPO/kino/packages/ltx-core/src"

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/samples" "$(dirname "$PHASE2_RUN")"

# vLLM ``start`` execs and blocks; never run it in the foreground inside this script.
_oneshot_stop_vllm() {
  if [[ "${VLLM_STARTED_BY_ONESHOT:-0}" != "1" ]]; then
    return 0
  fi
  echo "Stopping vLLM started by oneshot (port ${VLLM_PORT:-8001})…"
  python "$GOPEX_REPO/tools/vllm_local_server.py" stop --port "${VLLM_PORT:-8001}" 2>/dev/null \
    || pkill -f 'vllm_serve_launcher.py serve.*--port 8001' 2>/dev/null \
    || true
  VLLM_STARTED_BY_ONESHOT=0
}

_oneshot_vllm_ready() {
  curl -sf --max-time 5 "http://127.0.0.1:${VLLM_PORT:-8001}/v1/models" >/dev/null 2>&1
}

_start_vllm_background() {
  local port="${VLLM_PORT:-8001}"
  if _oneshot_vllm_ready; then
    echo "vLLM already listening on :${port}"
    return 0
  fi
  echo "Starting vLLM in background on :${port} (log: ${WORK_ROOT}/logs/vllm-8001.log)…"
  nohup python "$GOPEX_REPO/tools/vllm_local_server.py" start "${VLLM_PRESET:-gemma4_31b_it_local}" \
    --port "$port" >>"${WORK_ROOT}/logs/vllm-8001.log" 2>&1 &
  local i
  for i in $(seq 1 120); do
    if _oneshot_vllm_ready; then
      echo "vLLM ready on :${port}"
      VLLM_STARTED_BY_ONESHOT=1
      return 0
    fi
    sleep 5
  done
  echo "ERROR: vLLM did not become ready within 10 minutes. See ${WORK_ROOT}/logs/vllm-8001.log" >&2
  return 1
}

trap '_oneshot_stop_vllm' EXIT INT TERM

_run_batch_preprocess() {
  echo "=== Batch preprocess all projects → ${WORK_ROOT}/precomputed/ ==="
  if ! nvidia-smi >/dev/null 2>&1; then
    export GOPEX_FORCE_NVML_SAFE=1
    echo "nvidia-smi failed — GOPEX_FORCE_NVML_SAFE=1 (Gemma + embeddings processor on CPU where needed)"
  fi
  local _pre_root="$WORK_ROOT"
  local _pre_args=(
    --search-roots /run/media/tim/Datasets/datasets
    --run --preprocess-only --skip-existing --continue-on-error
    --work-root "$_pre_root"
    --model-path "$LTX_CKPT"
    --text-encoder-path "$GEMMA"
    --flat-dim-bridge-rank "$BRIDGE_RANK"
    --vae-tiling --load-text-encoder-in-8bit
  )
  if [[ "${GOPEX_HDR_PREPROCESS:-0}" == "1" ]]; then
    _pre_root="$WORK_ROOT"
    export GOPEX_PRECOMPUTED_SUBDIR="precomputed-hdr"
    _pre_args=(
      --search-roots /run/media/tim/Datasets/datasets
      --run --preprocess-only --skip-existing --continue-on-error
      --work-root "$_pre_root"
      --model-path "$LTX_CKPT"
      --text-encoder-path "$GEMMA"
      --flat-dim-bridge-rank "$BRIDGE_RANK"
      --vae-tiling --load-text-encoder-in-8bit
      --hdr-ingest
      --hdr-transfer "${HDR_TRANSFER:-auto}"
      --hdr-vae-encoding "${HDR_VAE_ENCODING:-logc3}"
      --hdr-synth-bracket-ev "${HDR_SYNTH_BRACKET_EV:--7:5:1}"
    )
    echo "HDR preprocess → ${WORK_ROOT}/precomputed-hdr/ (GOPEX_HDR_PREPROCESS=1)"
  else
    unset GOPEX_PRECOMPUTED_SUBDIR 2>/dev/null || true
  fi
  python "$GOPEX_REPO/tools/dataset_batch_pipeline.py" /run/media/tim/Expansion/datasets \
    "${_pre_args[@]}" \
    2>&1 | tee -a "$WORK_ROOT/logs/batch-preprocess.log"
  echo "=== Audit precomputed trees (strict) ==="
  for root in \
    "$PRECOMPUTED_ROOT" \
    "${PRECOMPUTED_HDR_ROOT:-}" \
    "$WORK_ROOT/precomputed" \
    "$WORK_ROOT/precomputed-hdr" \
    /run/media/tim/Expansion/gopex-ltx/precomputed \
    /run/media/tim/Expansion/gopex-ltx/gemma31b-r512/precomputed; do
    [[ -z "$root" ]] && continue
    if [[ -d "$root" ]]; then
      echo "--- audit: $root"
      python "$GOPEX_REPO/tools/ltx_precomputed_audit.py" --preprocessed-root "$root" --strict || true
    fi
  done
}

# PREPROCESS_ONLY=1 ./kino/oneshot.sh  — datasets only (no inference/vLLM/train/HQ)
if [[ "${PREPROCESS_ONLY:-0}" == "1" ]]; then
  _run_batch_preprocess
  echo "Preprocess batch done. Log: $WORK_ROOT/logs/batch-preprocess.log"
  exit 0
fi

echo "=== 1) Smoke inference (LoRA + connector sidecar + bridge) ==="
echo "Using LoRA: $LORA_CKPT"
test -f "$LORA_CKPT" || { echo "Missing $LORA_CKPT — wait for training or export LORA_CKPT=/path/to/lora_weights_step_XXXXX.safetensors"; exit 1; }
cd "$TRAINER"
python scripts/inference.py \
  --checkpoint "$LTX_CKPT" \
  --text-encoder-path "$GEMMA" \
  --lora-path "$LORA_CKPT" \
  --flat-dim-bridge-rank "$BRIDGE_RANK" \
  --prompt "$PROMPT" \
  --negative-prompt "worst quality, inconsistent motion, blurry, jittery, distorted" \
  --height 576 --width 1024 --num-frames 41 --frame-rate 24 \
  --num-inference-steps 30 --guidance-scale 4.0 \
  --output "$INFER_OUT"

if [[ "${RUN_YOUTUBE_MANIFEST:-0}" == "1" ]]; then
  echo "=== 2–3) vLLM + youtube-public-domain manifest (optional; needs GPU 0 free) ==="
  _start_vllm_background
  RUN=1 bash "$GOPEX_REPO/scripts/run-youtube-public-domain-manifest.sh" \
    2>&1 | tee -a "$WORK_ROOT/logs/youtube-public-domain-manifest.log"
  _oneshot_stop_vllm
else
  echo "=== 2–3) Skipping vLLM / youtube manifest (set RUN_YOUTUBE_MANIFEST=1 to enable) ==="
fi

echo "=== 4–5) Preprocess all datasets + audit ==="
_run_batch_preprocess

if [[ "${RUN_PHASE2_TRAIN:-0}" == "1" ]]; then
  echo "=== 6) Phase 2 LoRA (merged precompute, warm-start 6k LoRA) ==="
  bash "$GOPEX_REPO/scripts/kino-phase2-train.sh"
else
  echo "=== 6) Skipping Phase 2 train (./scripts/kino-phase2-train.sh or RUN_PHASE2_TRAIN=1) ==="
fi

# ── OPTIONAL: Phase 1a text stack (bridge + aggregates, frozen DiT) ─────────
# Stop vLLM; needs dataset.json on PRECOMPUTED_ROOT's project. Uncomment to run:
# python scripts/train.py configs/ltx2_text_stack_gemma4_bridge.yaml \
#   2>&1 | tee "$WORK_ROOT/logs/phase1a-text-stack.log"
# TEXT_STACK="$(ls -t ${WORK_ROOT}/runs/*/checkpoints/text_stack_weights_step_*.safetensors 2>/dev/null | head -1)"
# NATIVE_LTX="${WORK_ROOT}/models/ltx-2.3-22b-gemma4-native.safetensors"
# python scripts/fold_flat_dim_bridge.py \
#   --ltx "$LTX_CKPT" --gemma "$GEMMA" --text-stack "$TEXT_STACK" \
#   --output "$NATIVE_LTX" --flat-dim-bridge-rank "$BRIDGE_RANK"
# python "$GOPEX_REPO/tools/ltx_gemma_flat_probe.py" --ltx "$NATIVE_LTX" --gemma "$GEMMA"
# Re-preprocess with native LTX after fold, then re-run step 6 with PRECOMPUTED_ROOT updated.

if [[ "${RUN_HQ:-0}" != "1" ]]; then
  echo "=== 7) Skipping HQ two-stage (set RUN_HQ=1 when GPU is free) ==="
  echo "=== 8) Qt UIs (optional, same assets) ==="
  echo "  python $GOPEX_REPO/tools/two_stage_hq_kino_qt.py"
  echo "  python $GOPEX_REPO/tools/ltx_trainer_qt.py"
  echo "  python $GOPEX_REPO/tools/dataset_pipeline_qt.py"
  echo "Done. Smoke: $INFER_OUT"
  exit 0
fi

echo "=== 7) HQ two-stage delivery (stop training; ~80GB+ free on GPU 0) ==="
export LTX_EXPERIMENTAL_ENCODE_FLAT_BRIDGE=1
python "$GOPEX_REPO/pipeline/two_stage_hq_kino.py" \
  --checkpoint-path "$LTX_CKPT" \
  --gemma-root "$GEMMA" \
  --prompt "$PROMPT" \
  --output-path "$HQ_OUT" \
  --distilled-lora "$DISTILLED_LORA" 1.0 \
  --spatial-upsampler-path "$UPSAMPLER" \
  --lora "$LORA_CKPT" 0.85 \
  --seed 42 \
  --height 1088 --width 1920 --num-frames 121 --frame-rate 24 \
  --num-inference-steps 15 \
  --distilled-lora-strength-stage-1 0.25 \
  --distilled-lora-strength-stage-2 0.5 \
  --experimental-flat-dim-bridge \
  2>&1 | tee "$WORK_ROOT/logs/hq-two-stage.log"

if [[ "${RUN_SFM_4DGS:-0}" == "1" && -f "$HQ_OUT" ]]; then
  echo "=== 7b) SfM → 4DGS bundle from HQ output ==="
  SFM_4DGS_DIR="${WORK_ROOT}/sfm/hq_two_stage"
  python "$GOPEX_REPO/tools/sfm_to_4dgs.py" --video "$HQ_OUT" --output-dir "$SFM_4DGS_DIR" --backend auto \
    2>&1 | tee "$WORK_ROOT/logs/sfm-4dgs.log"
  python "$GOPEX_REPO/tools/sfm_to_4dgs.py" --sfm-dir "$SFM_4DGS_DIR" --plan
fi

echo "=== 8) Qt UIs (optional, same assets) ==="
echo "  python $GOPEX_REPO/tools/two_stage_hq_kino_qt.py"
echo "  python $GOPEX_REPO/tools/ltx_trainer_qt.py"
echo "  python $GOPEX_REPO/tools/dataset_pipeline_qt.py"
echo "Done. Smoke: $INFER_OUT  HQ: $HQ_OUT"
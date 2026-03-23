#!/usr/bin/env bash
set -euo pipefail

# Nyaymalaw RunPod launcher for Qwen3-8B fine-tuning (48 GB VRAM profile).
# Run from repo root:
#   bash training/runpod_finetune_qwen3_8b.sh

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. This script must run on a GPU pod."
  exit 1
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Start a GPU pod and try again.")
print("GPU:", torch.cuda.get_device_name(0))
print("Total VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1))
PY

export PYTHONIOENCODING=utf-8
export NYAYMALAW_MODEL_NAME="${NYAYMALAW_MODEL_NAME:-unsloth/Qwen3-8B-bnb-4bit}"
export NYAYMALAW_VRAM_TARGET_GB="${NYAYMALAW_VRAM_TARGET_GB:-48}"
export NYAYMALAW_MAX_SEQ_LEN="${NYAYMALAW_MAX_SEQ_LEN:-2048}"
export NYAYMALAW_BATCH_SIZE="${NYAYMALAW_BATCH_SIZE:-8}"
export NYAYMALAW_GRAD_ACCUM="${NYAYMALAW_GRAD_ACCUM:-2}"
export NYAYMALAW_EVAL_BATCH="${NYAYMALAW_EVAL_BATCH:-4}"
export NYAYMALAW_LORA_RANK="${NYAYMALAW_LORA_RANK:-64}"
export NYAYMALAW_LORA_ALPHA="${NYAYMALAW_LORA_ALPHA:-128}"
export NYAYMALAW_EPOCHS="${NYAYMALAW_EPOCHS:-3}"
export NYAYMALAW_LR="${NYAYMALAW_LR:-2e-4}"
export NYAYMALAW_WARMUP_RATIO="${NYAYMALAW_WARMUP_RATIO:-0.05}"
export NYAYMALAW_WEIGHT_DECAY="${NYAYMALAW_WEIGHT_DECAY:-0.01}"
export NYAYMALAW_SAVE_STEPS="${NYAYMALAW_SAVE_STEPS:-25}"
export NYAYMALAW_DATALOADER_WORKERS="${NYAYMALAW_DATALOADER_WORKERS:-4}"
export NYAYMALAW_REPORT_TO="${NYAYMALAW_REPORT_TO:-none}"
export NYAYMALAW_GRADIENT_CHECKPOINTING="${NYAYMALAW_GRADIENT_CHECKPOINTING:-true}"

mkdir -p training
echo "Starting fine-tune. Logs: training/runpod_train.log"
python training/finetune_unsloth.py 2>&1 | tee training/runpod_train.log

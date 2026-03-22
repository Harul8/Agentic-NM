"""
Nyaymalaw — Unsloth QLoRA fine-tune script
Base model  : Qwen3-8B (thinking mode — silent CoT improves legal reasoning)
Hardware    : RTX 4060 8 GB VRAM

Run from the project root:
    cd "Nyaymalaw 4.0"
    python training/finetune_unsloth.py

Outputs:
    training/lora_model/          ← LoRA adapter weights (checkpoint)
    training/merged_model/        ← Full merged fp16 model (for GGUF export)

── Thinking mode note ────────────────────────────────────────────────────────
Qwen3 supports an internal chain-of-thought enclosed in <think>…</think> tags.
During fine-tuning we keep thinking ENABLED (no /no_think token injected).
This lets the model reason silently about forum, urgency, and decision dimensions
before producing its visible reply — matching how a senior advocate actually works.
At inference time you can suppress the <think> block from the user-visible output
by stripping everything between <think> and </think> in your API server.
──────────────────────────────────────────────────────────────────────────────
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # reduce fragmentation
# Avoid Unsloth fused CE VRAM probe crashes on 8GB cards / fragmented VRAM.
# Must be set before importing unsloth.
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

from unsloth import FastLanguageModel
import unsloth.models.llama as unsloth_llama
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
import torch, pathlib, subprocess, time, gc
import torch.nn.functional as F


def _safe_ce_fallback(
    trainer,
    hidden_states,
    lm_head_weight,
    lm_head_bias,
    labels,
    mask=None,
    n_items=None,
    scaling=None,
    target_gb=None,
    torch_compile=True,
    logit_softcapping=0,
    **kwargs,
):
    """
    Fallback for environments where Unsloth fused CE fails VRAM probing.
    Uses standard shifted-token cross entropy.
    """
    del trainer, mask, target_gb, torch_compile, kwargs
    logits = F.linear(hidden_states, lm_head_weight, lm_head_bias)
    if logit_softcapping and logit_softcapping > 0:
        logits = logit_softcapping * torch.tanh(logits / logit_softcapping)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    if n_items is not None:
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_items = n_items.to(dtype=torch.float32) if torch.is_tensor(n_items) else torch.tensor(float(n_items), device=loss.device)
        loss = loss / torch.clamp(n_items, min=1.0)
    else:
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )

    return loss * scaling if scaling is not None else loss


# Hard-disable fused CE path that errors with "No or negligible GPU memory..."
unsloth_llama.unsloth_fused_ce_loss = _safe_ce_fallback

# ── 1. Config ─────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).resolve().parent          # .../training/
DATA_DIR   = BASE_DIR / "finetune_ready"
OUTPUT_DIR = BASE_DIR / "lora_model"
MERGE_DIR  = BASE_DIR / "merged_model"

# Qwen3-4B — 4-bit quantised via Unsloth (~2.8 GB, leaves ~5 GB free for training)
# 8B OOMs on RTX 4060 8GB due to Qwen3's 151k-token vocab CE loss buffer requirement.
MODEL_NAME = "unsloth/Qwen3-4B-bnb-4bit"
# Fallback:  "unsloth/Qwen3-4B-Instruct-bnb-4bit"
# Fallback2: "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

MAX_SEQ_LEN   = 1024   # VRAM headroom; ~19/1034 train examples exceed 1024 tokens (will truncate)
LORA_RANK     = 16     # reduced from 32 — saves ~40 MB adapter memory
LORA_ALPHA    = 32     # alpha = 2 × rank
EVAL_SUBSET_SIZE = 48  # lighter eval (VRAM + time); still epoch-aligned, seeded shuffle
LORA_DROPOUT  = 0.05
BATCH_SIZE    = 1      # reduced from 2 — halves activation memory on 8 GB
EVAL_BATCH    = 1      # eval uses full-length examples — stay at 1 to avoid OOM
GRAD_ACCUM    = 16     # effective batch = 1 × 16 = 16 (same as before)
EPOCHS        = 3
LR            = 2e-4
WARMUP_RATIO  = 0.05
WEIGHT_DECAY  = 0.01
SEED          = 42

# Prefer bf16 on capable GPUs (Ada, etc.). If bitsandbytes backward hits
# CUBLAS_STATUS_EXECUTION_FAILED (often Windows + 4-bit), set NYAYMALAW_FP16=1.
_USE_FP16 = os.environ.get("NYAYMALAW_FP16", "0") == "1"

# ── Thermal throttle settings ─────────────────────────────────────────────────
TEMP_LIMIT_C  = 80     # pause training above this GPU temperature
COOL_WAIT_S   = 30     # seconds to idle before re-checking temperature
MIN_FREE_VRAM_MB = 700 # pause step if free VRAM drops below this

# ── 2. Load model + tokeniser ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Loading base model : {MODEL_NAME}")
print(f"  Thinking mode      : ENABLED (silent CoT)")
if _USE_FP16:
    _compute_dtype = torch.float16
elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    _compute_dtype = torch.bfloat16
else:
    _compute_dtype = torch.float16
_dtype_label = "fp16 (NYAYMALAW_FP16=1)" if _USE_FP16 else str(_compute_dtype).split(".")[-1]
print(f"  Compute dtype      : {_dtype_label}")
print(f"{'='*60}\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LEN,
    dtype           = _compute_dtype,
    load_in_4bit    = True,           # 4-bit NF4 weights — essential for 8 GB
    # Unsloth enables double-quantisation internally when load_in_4bit=True,
    # saving an extra ~0.4 GB vs single-quant.
)

# ── 3. Attach LoRA adapters ───────────────────────────────────────────────────
# Qwen3 uses the same projection names as Qwen2 — no changes needed here.
model = FastLanguageModel.get_peft_model(
    model,
    r                   = LORA_RANK,
    target_modules      = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha          = LORA_ALPHA,
    lora_dropout        = LORA_DROPOUT,
    bias                = "none",
    use_gradient_checkpointing = "unsloth",   # saves ~30% VRAM
    random_state        = SEED,
    use_rslora          = False,
    loftq_config        = None,
)

# ── 4. Thermal throttle callback ─────────────────────────────────────────────
class ThermalThrottleCallback(TrainerCallback):
    """
    Pauses training at the start of each step if the GPU temperature exceeds
    TEMP_LIMIT_C degrees.  Idles in COOL_WAIT_S-second bursts until the GPU
    cools back below the limit before allowing the step to proceed.

    Temperature is read via pynvml (preferred) or nvidia-smi (fallback).
    If neither is available the callback is a silent no-op.
    """

    def __init__(self, max_temp: int = TEMP_LIMIT_C, wait_s: int = COOL_WAIT_S):
        self.max_temp = max_temp
        self.wait_s   = wait_s

    def _gpu_temp(self) -> int:
        """Return current GPU 0 temperature in °C, or 0 if unreadable."""
        # ── Try pynvml first (zero-overhead, no subprocess) ───────────────────
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            pass
        # ── Fallback: nvidia-smi subprocess ───────────────────────────────────
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            return int(out.stdout.strip())
        except Exception:
            return 0   # can't read — don't throttle

    def on_step_begin(self, args, state, control, **kwargs):
        temp = self._gpu_temp()
        if temp == 0 or temp <= self.max_temp:
            return                           # all clear — proceed immediately

        print(f"\n🌡  GPU {temp}°C > {self.max_temp}°C — pausing training. "
              f"Checking every {self.wait_s}s …")
        while temp > self.max_temp:
            time.sleep(self.wait_s)
            temp = self._gpu_temp()
            if temp > self.max_temp:
                print(f"   Still {temp}°C — cooling down …")
            else:
                print(f"   GPU cooled to {temp}°C — resuming.\n")


class VramGuardCallback(TrainerCallback):
    """
    Pauses at step start if currently free VRAM is below a safety floor.
    Helps avoid mid-run OOM spikes on near-capacity 8GB GPUs.
    """

    def __init__(self, min_free_mb: int = MIN_FREE_VRAM_MB, wait_s: int = 10):
        self.min_free_mb = min_free_mb
        self.wait_s = wait_s

    @staticmethod
    def _free_vram_mb() -> int:
        if not torch.cuda.is_available():
            return 0
        free_bytes, _ = torch.cuda.mem_get_info(0)
        return int(free_bytes / 1024 / 1024)

    def on_step_begin(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return
        free_mb = self._free_vram_mb()
        if free_mb >= self.min_free_mb:
            return

        print(f"\n🧯  Low free VRAM: {free_mb} MB < {self.min_free_mb} MB — pausing.")
        while free_mb < self.min_free_mb:
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(self.wait_s)
            free_mb = self._free_vram_mb()
            if free_mb < self.min_free_mb:
                print(f"   Free VRAM still low: {free_mb} MB — waiting...")
        print(f"   Free VRAM recovered: {free_mb} MB — resuming.\n")

# ── 5. Load datasets ──────────────────────────────────────────────────────────
print("Loading train / val datasets …")  # ── was §4, renumbered below
train_ds = load_dataset("json", data_files=str(DATA_DIR / "train.jsonl"), split="train")
val_ds   = load_dataset("json", data_files=str(DATA_DIR / "val.jsonl"),   split="train")
print(f"  Train: {len(train_ds)} examples")
val_full_n = len(val_ds)
if val_full_n > EVAL_SUBSET_SIZE:
    val_ds = val_ds.shuffle(seed=SEED).select(range(EVAL_SUBSET_SIZE))
print(f"  Val:   {len(val_ds)} examples (subset of {val_full_n} from val.jsonl)\n")

# ── 5. Format into Qwen3 chat template ───────────────────────────────────────
# Qwen3's apply_chat_template accepts an optional enable_thinking kwarg.
# Passing enable_thinking=True (the default) preserves the <think> blocks in
# training targets so the model learns to reason before answering.
def format_example(example):
    """Convert messages list → single tokenised string using Qwen3 chat template."""
    try:
        # Qwen3 tokenizer supports enable_thinking
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize         = False,
            add_generation_prompt = False,
            enable_thinking  = True,
        )
    except TypeError:
        # Older tokenizer version — fall back to standard call
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize    = False,
            add_generation_prompt = False,
        )
    return {"text": text}

train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(format_example,   remove_columns=val_ds.column_names)

# ── 6. Training arguments ─────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir                  = str(OUTPUT_DIR),
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    per_device_eval_batch_size  = EVAL_BATCH,   # 1 — eval examples can be long
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    learning_rate               = LR,
    weight_decay                = WEIGHT_DECAY,
    warmup_ratio                = WARMUP_RATIO,
    lr_scheduler_type           = "cosine",
    fp16                        = _USE_FP16 or not torch.cuda.is_bf16_supported(),
    bf16                        = not _USE_FP16 and torch.cuda.is_bf16_supported(),
    logging_steps               = 20,
    report_to                   = "none",       # swap to "wandb" if you want tracking
    seed                        = SEED,
    optim                       = "adamw_8bit", # Unsloth 8-bit Adam — saves VRAM
    dataloader_num_workers      = 0,            # avoids multiprocessing issues on Windows
    torch_empty_cache_steps     = 20,           # periodically release cached VRAM to reduce spikes
)

# ── 7. Trainer ────────────────────────────────────────────────────────────────
thermal_cb = ThermalThrottleCallback(max_temp=TEMP_LIMIT_C, wait_s=COOL_WAIT_S)
vram_cb = VramGuardCallback(min_free_mb=MIN_FREE_VRAM_MB, wait_s=10)
print(f"  Thermal throttle   : pause if GPU > {TEMP_LIMIT_C}°C, "
      f"idle {COOL_WAIT_S}s per check\n")
print(f"  VRAM guard         : pause if free VRAM < {MIN_FREE_VRAM_MB} MB\n")

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_ds,
    eval_dataset       = val_ds,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    data_collator      = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    args               = training_args,
    callbacks          = [thermal_cb, vram_cb],
)

# ── 8. Train ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Starting training  —  {EPOCHS} epochs, effective batch {BATCH_SIZE * GRAD_ACCUM}")
print(f"{'='*60}\n")

# Free any fragmented/reserved-but-idle VRAM before the first forward pass
gc.collect()
torch.cuda.empty_cache()

# Unsloth compiler can change this during setup; enforce right before train loop.
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
trainer_stats = trainer.train()

print(f"\nTraining complete.")
print(f"  Final train loss : {trainer_stats.training_loss:.4f}")

# ── 9. Save LoRA adapter ──────────────────────────────────────────────────────
print(f"\nSaving LoRA adapter → {OUTPUT_DIR}")
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

# ── 10. Merge + save full model (needed for GGUF export) ─────────────────────
print(f"\nMerging LoRA → full fp16 model → {MERGE_DIR}")
print("(This uses extra VRAM; if OOM, run merge_and_export.py separately after rebooting.)\n")

try:
    model.save_pretrained_merged(
        str(MERGE_DIR),
        tokenizer,
        save_method = "merged_16bit",
    )
    print("Merge complete.")
except Exception as e:
    print(f"Merge skipped (OOM or error): {e}")
    print("Run training/merge_and_export.py separately to merge + export GGUF.")

print("\nDone. Next step: run training/merge_and_export.py to produce the GGUF file for Ollama.")

"""
training/finetune.py — Unsloth QLoRA fine-tune script.
"""
# ── 0. Imports ────────────────────────────────────────────────────────────────
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # reduce fragmentation
_DISABLE_FUSED_CE = os.environ.get("NYAYMALAW_DISABLE_FUSED_CE", "1") == "1"
if _DISABLE_FUSED_CE:
    # Force fallback CE path (returns logits) to avoid fused CE workspace spikes.
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
from peft import PeftModel
import sys
import torch, pathlib, subprocess, time, gc

# ── 1. Config ─────────────────────────────────────────────────────────────────
# Profile: ~48 GB VRAM — larger batches, longer context headroom, full AdamW.
VRAM_TARGET_GB = int(os.environ.get("NYAYMALAW_VRAM_TARGET_GB", "48"))
BASE_DIR   = pathlib.Path(__file__).resolve().parent          # .../training/
DATA_DIR   = pathlib.Path(os.environ.get("NYAYMALAW_DATA_DIR", str(BASE_DIR / "finetune_ready")))
OUTPUT_DIR = pathlib.Path(os.environ.get("NYAYMALAW_OUTPUT_DIR", str(BASE_DIR / "lora_model")))
MERGE_DIR  = pathlib.Path(os.environ.get("NYAYMALAW_MERGE_DIR", str(BASE_DIR / "merged_model")))

# ── How to continue training (pick ONE) ───────────────────────────────────────
# LOAD_ADAPTER_FROM — path to checkpoint-* with LoRA weights. Loads adapter + fresh Adam.
# RESUME_FROM — full Trainer resume (optimizer + scheduler + step counter).
_load_adapter_env = os.environ.get("NYAYMALAW_LOAD_ADAPTER_FROM", "").strip()
LOAD_ADAPTER_FROM = pathlib.Path(_load_adapter_env) if _load_adapter_env else None
_resume_env = os.environ.get("NYAYMALAW_RESUME_FROM", "").strip()
RESUME_FROM = pathlib.Path(_resume_env) if _resume_env else None
if LOAD_ADAPTER_FROM and RESUME_FROM:
    raise ValueError(
        "Set only one of NYAYMALAW_LOAD_ADAPTER_FROM or NYAYMALAW_RESUME_FROM, not both."
    )

# Qwen3-8B — 4-bit via Unsloth; ~48 GB fits higher batch + rank + seq than 8–24 GB setups.
MODEL_NAME = os.environ.get("NYAYMALAW_MODEL_NAME", "unsloth/Qwen3-8B-bnb-4bit")
# Fallback:  "unsloth/Qwen3-8B-Instruct-bnb-4bit"

MAX_SEQ_LEN      = int(os.environ.get("NYAYMALAW_MAX_SEQ_LEN", "1596"))
LORA_RANK        = int(os.environ.get("NYAYMALAW_LORA_RANK", "32"))
LORA_ALPHA       = int(os.environ.get("NYAYMALAW_LORA_ALPHA", "64"))
EVAL_SUBSET_SIZE = int(os.environ.get("NYAYMALAW_EVAL_SUBSET_SIZE", "145"))
LORA_DROPOUT     = float(os.environ.get("NYAYMALAW_LORA_DROPOUT", "0.0"))
BATCH_SIZE       = int(os.environ.get("NYAYMALAW_BATCH_SIZE", "4"))
EVAL_BATCH       = int(os.environ.get("NYAYMALAW_EVAL_BATCH", "1"))
GRAD_ACCUM       = int(os.environ.get("NYAYMALAW_GRAD_ACCUM", "4"))
SEED             = int(os.environ.get("NYAYMALAW_SEED", "42"))

# Scratch run defaults (used unless overridden by NYAYMALAW_* env vars)
SCRATCH_EPOCHS         = int(os.environ.get("NYAYMALAW_SCRATCH_EPOCHS", "4"))
SCRATCH_LR             = float(os.environ.get("NYAYMALAW_SCRATCH_LR", "2e-4"))
SCRATCH_WEIGHT_DECAY   = float(os.environ.get("NYAYMALAW_SCRATCH_WEIGHT_DECAY", "0.01"))
SCRATCH_LR_SCHEDULER   = os.environ.get("NYAYMALAW_SCRATCH_LR_SCHEDULER", "cosine")
SCRATCH_WARMUP_STEPS   = int(os.environ.get("NYAYMALAW_SCRATCH_WARMUP_STEPS", "20"))

EPOCHS = int(os.environ.get("NYAYMALAW_EPOCHS", str(SCRATCH_EPOCHS)))
LR = float(os.environ.get("NYAYMALAW_LR", str(SCRATCH_LR)))
WEIGHT_DECAY = float(os.environ.get("NYAYMALAW_WEIGHT_DECAY", str(SCRATCH_WEIGHT_DECAY)))
LR_SCHEDULER = os.environ.get("NYAYMALAW_LR_SCHEDULER", SCRATCH_LR_SCHEDULER)
WARMUP_STEPS = int(os.environ.get("NYAYMALAW_WARMUP_STEPS", str(SCRATCH_WARMUP_STEPS)))

# Prefer bf16 on capable GPUs (Ada, etc.). If bitsandbytes backward hits
# CUBLAS_STATUS_EXECUTION_FAILED (often Windows + 4-bit), set NYAYMALAW_FP16=1.
_USE_FP16 = os.environ.get("NYAYMALAW_FP16", "0") == "1"

# ── Thermal throttle settings ─────────────────────────────────────────────────
TEMP_LIMIT_C  = int(os.environ.get("NYAYMALAW_TEMP_LIMIT_C", "80"))
COOL_WAIT_S   = int(os.environ.get("NYAYMALAW_COOL_WAIT_S", "30"))
MIN_FREE_VRAM_MB = int(os.environ.get("NYAYMALAW_MIN_FREE_VRAM_MB", "256"))
REPORT_TO = os.environ.get("NYAYMALAW_REPORT_TO", "none")
SAVE_STEPS = int(os.environ.get("NYAYMALAW_SAVE_STEPS", "25"))
DATALOADER_WORKERS = int(
    os.environ.get("NYAYMALAW_DATALOADER_WORKERS", "4" if sys.platform != "win32" else "0")
)
DISABLE_THERMAL_GUARD = os.environ.get("NYAYMALAW_DISABLE_THERMAL_GUARD", "0") == "1"
DISABLE_VRAM_GUARD = os.environ.get("NYAYMALAW_DISABLE_VRAM_GUARD", "0") == "1"
GRADIENT_CHECKPOINTING = os.environ.get("NYAYMALAW_GRADIENT_CHECKPOINTING", "true").lower() == "true"
DEBUG_VRAM = os.environ.get("NYAYMALAW_DEBUG_VRAM", "0") == "1"

# ── 2. Load model + tokeniser ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Loading base model : {MODEL_NAME}")
print(f"  VRAM profile       : ~{VRAM_TARGET_GB} GB (batch {BATCH_SIZE} × accum {GRAD_ACCUM}, seq {MAX_SEQ_LEN})")
print(f"  Thinking mode      : ENABLED (silent CoT)")
if LOAD_ADAPTER_FROM:
    print(f"  Continue mode      : LOAD_ADAPTER_FROM={LOAD_ADAPTER_FROM}")
elif RESUME_FROM:
    print(f"  Continue mode      : RESUME_FROM={RESUME_FROM}")
else:
    print("  Continue mode      : SCRATCH")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected. Run this on a GPU pod (RunPod) with CUDA enabled.")
gpu_name = torch.cuda.get_device_name(0)
gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
print(f"  GPU                : {gpu_name} ({gpu_total_gb:.1f} GB)")
if _USE_FP16:
    _compute_dtype = torch.float16
elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    _compute_dtype = torch.bfloat16
else:
    _compute_dtype = torch.float16
_dtype_label = "fp16 (NYAYMALAW_FP16=1)" if _USE_FP16 else str(_compute_dtype).split(".")[-1]
print(f"  Compute dtype      : {_dtype_label}")
if _DISABLE_FUSED_CE:
    print("  CE path            : fallback (UNSLOTH_RETURN_LOGITS=1)")
else:
    print("  CE path            : fused (default)")
if LORA_DROPOUT > 0:
    print(
        f"  LoRA dropout       : {LORA_DROPOUT} (disables Unsloth fast LoRA patching; may increase VRAM use)"
    )
else:
    print(f"  LoRA dropout       : {LORA_DROPOUT} (enables Unsloth fast LoRA patching)")
print(f"{'='*60}\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LEN,
    dtype           = _compute_dtype,
    load_in_4bit    = True,           # 4-bit NF4 weights
    # Unsloth enables double-quantisation internally when load_in_4bit=True,
    # saving an extra ~0.4 GB vs single-quant.
)

# ── 3. Attach LoRA adapters ───────────────────────────────────────────────────
# Qwen3 uses the same projection names as Qwen2 — no changes needed here.
if LOAD_ADAPTER_FROM:
    print(f"Loading existing LoRA adapter from: {LOAD_ADAPTER_FROM}")
    model = PeftModel.from_pretrained(model, str(LOAD_ADAPTER_FROM), is_trainable=True)
else:
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
        use_gradient_checkpointing = "unsloth" if GRADIENT_CHECKPOINTING else False,
        random_state        = SEED,
        use_rslora          = False,
        loftq_config        = None,
    )

# Reduce training-time memory pressure and print checkpointing state explicitly.
if hasattr(model, "config"):
    model.config.use_cache = False
gc_state = bool(getattr(model, "is_gradient_checkpointing", False))
print(f"  Gradient checkpointing active: {gc_state}")

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
    Helps avoid mid-run OOM spikes when free VRAM is critically low.
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


class VramDebugCallback(TrainerCallback):
    """Prints periodic CUDA memory telemetry for peak usage debugging."""

    def __init__(self, every_steps: int = 10):
        self.every_steps = max(1, every_steps)

    @staticmethod
    def _mb(value_bytes: int) -> int:
        return int(value_bytes / 1024 / 1024)

    def on_step_end(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return
        if state.global_step <= 0 or (state.global_step % self.every_steps) != 0:
            return

        allocated = self._mb(torch.cuda.memory_allocated(0))
        reserved = self._mb(torch.cuda.memory_reserved(0))
        peak_alloc = self._mb(torch.cuda.max_memory_allocated(0))
        peak_reserved = self._mb(torch.cuda.max_memory_reserved(0))
        free_mb, total_mb = torch.cuda.mem_get_info(0)
        free_mb = self._mb(free_mb)
        total_mb = self._mb(total_mb)
        print(
            f"🧠 VRAM step {state.global_step}: "
            f"alloc={allocated}MB reserved={reserved}MB "
            f"peak_alloc={peak_alloc}MB peak_reserved={peak_reserved}MB "
            f"free={free_mb}MB/{total_mb}MB"
        )
        torch.cuda.reset_peak_memory_stats(0)

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
    eval_strategy               = "steps",
    eval_steps                  = SAVE_STEPS,
    save_strategy               = "steps",   # spot-safe: checkpoint every 25 steps
    save_steps                  = SAVE_STEPS,
    save_total_limit            = 4,         # keep last 4 checkpoints only
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    learning_rate               = LR,
    weight_decay                = WEIGHT_DECAY,
    # label_smoothing_factor removed — when > 0, HF Trainer bypasses Unsloth's
    # chunked fused CE and materialises full fp32 logits (151k vocab × batch × seq),
    # causing ~45 GB VRAM use and numerically unstable loss on 4-bit models.
    warmup_steps                = WARMUP_STEPS,
    lr_scheduler_type           = LR_SCHEDULER,
    fp16                        = _USE_FP16 or not torch.cuda.is_bf16_supported(),
    bf16                        = not _USE_FP16 and torch.cuda.is_bf16_supported(),
    logging_steps               = 20,
    report_to                   = REPORT_TO,
    seed                        = SEED,
    optim                       = "adamw_torch",  # 48 GB: full-precision Adam (8-bit optional on tight VRAM)
    dataloader_num_workers      = DATALOADER_WORKERS,
    torch_empty_cache_steps     = 50,
)

# ── 7. Trainer ────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MERGE_DIR.mkdir(parents=True, exist_ok=True)

callbacks = []
if not DISABLE_THERMAL_GUARD:
    callbacks.append(ThermalThrottleCallback(max_temp=TEMP_LIMIT_C, wait_s=COOL_WAIT_S))
    print(f"  Thermal throttle   : pause if GPU > {TEMP_LIMIT_C}C, idle {COOL_WAIT_S}s per check")
else:
    print("  Thermal throttle   : disabled by env")

if not DISABLE_VRAM_GUARD:
    callbacks.append(VramGuardCallback(min_free_mb=MIN_FREE_VRAM_MB, wait_s=10))
    print(f"  VRAM guard         : pause if free VRAM < {MIN_FREE_VRAM_MB} MB\n")
else:
    print("  VRAM guard         : disabled by env\n")
if DEBUG_VRAM:
    callbacks.append(VramDebugCallback(every_steps=10))
    print("  VRAM debug         : enabled (prints every 10 steps)\n")

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_ds,
    eval_dataset       = val_ds,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    data_collator      = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    args               = training_args,
    callbacks          = callbacks,
)

# ── 8. Train ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Starting training  —  {EPOCHS} epochs, effective batch {BATCH_SIZE * GRAD_ACCUM}")
print(f"{'='*60}\n")

# Free any fragmented/reserved-but-idle VRAM before the first forward pass
gc.collect()
torch.cuda.empty_cache()

_resume = str(RESUME_FROM) if (RESUME_FROM and RESUME_FROM.exists()) else None
if _resume:
    print(f"  Resuming from checkpoint : {_resume}\n")
else:
    print(f"  Fresh run — training from base model weights\n")

trainer_stats = trainer.train(resume_from_checkpoint=_resume)

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

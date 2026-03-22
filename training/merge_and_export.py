"""
Nyaymalaw — Post-training merge + GGUF export
Run this after finetune_unsloth.py completes (or if merge step was skipped due to OOM).

Requirements (run once in your ai-gpu venv):
    pip install llama-cpp-python   (for GGUF conversion, or use llama.cpp directly)

Easier alternative — use llama.cpp's convert script:
    python llama.cpp/convert_hf_to_gguf.py training/merged_model --outtype q4_k_m \
        --outfile training/nyaymalaw-q4_k_m.gguf
# Note: base model is Qwen3-4B (not 8B) — GGUF will be ~2.5 GB at q4_k_m

Then register with Ollama:
    ollama create nyaymalaw -f training/Modelfile
"""

from unsloth import FastLanguageModel
import pathlib, sys

BASE_DIR   = pathlib.Path(__file__).resolve().parent
LORA_DIR   = BASE_DIR / "lora_model"
MERGE_DIR  = BASE_DIR / "merged_model"
GGUF_PATH  = BASE_DIR / "nyaymalaw-q4_k_m.gguf"

# ── Step 1: Reload adapter and merge ─────────────────────────────────────────
print("Loading LoRA adapter …")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = str(LORA_DIR),
    max_seq_length = 2048,
    dtype          = None,
    load_in_4bit   = True,
)

print(f"Merging → {MERGE_DIR} …")
model.save_pretrained_merged(
    str(MERGE_DIR),
    tokenizer,
    save_method = "merged_16bit",
)
print("Merge complete.\n")

# ── Step 2: Export GGUF ───────────────────────────────────────────────────────
# Unsloth can export directly to GGUF (q4_k_m or q8_0)
print(f"Exporting GGUF (q4_k_m) → {GGUF_PATH} …")
print("(q4_k_m = ~5 GB, excellent quality/size balance for 8B models)\n")

try:
    model.save_pretrained_gguf(
        str(BASE_DIR / "nyaymalaw"),   # prefix; Unsloth appends quantisation tag
        tokenizer,
        quantization_method = "q4_k_m",
    )
    print("GGUF export complete.")
except Exception as e:
    print(f"Unsloth GGUF export failed: {e}")
    print("\nFallback: use llama.cpp manually:")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {MERGE_DIR} \\")
    print(f"    --outtype q4_k_m --outfile {GGUF_PATH}")
    sys.exit(1)

# ── Step 3: Create Ollama Modelfile ──────────────────────────────────────────
gguf_files = list(BASE_DIR.glob("nyaymalaw*.gguf"))
gguf_file  = gguf_files[0] if gguf_files else GGUF_PATH

modelfile = f"""\
FROM {gguf_file}

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

# Qwen3 thinking mode: set to "enabled" to keep <think> blocks,
# or "disabled" (/no_think) for faster responses without silent CoT.
PARAMETER thinking enabled

SYSTEM \"\"\"
You are a senior advocate at Nyaymalaw — an AI-powered Indian legal guidance platform.
Your role is to conduct a structured intake conversation with clients or junior advocates,
diagnose the matter's legal shape, identify the correct forums, and provide practical
next-step guidance grounded in Indian law.

During intake: ask one high-value diagnostic question at a time. Do not cite section
numbers or case names from memory — grounded citations come only after retrieval.
\"\"\"
"""

modelfile_path = BASE_DIR / "Modelfile"
modelfile_path.write_text(modelfile)
print(f"\nModelfile written → {modelfile_path}")
print("\nRegister with Ollama:")
print(f"  ollama create nyaymalaw -f {modelfile_path}")
print(f"  ollama run nyaymalaw")

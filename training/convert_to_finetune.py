"""
Convert Nyaymalaw training examples → Unsloth / LoRA fine-tuning format
========================================================================

Usage:
    python training/convert_to_finetune.py

Outputs two files in training/finetune_ready/:
    intake_finetune.jsonl   — intake conversation turns (fact-collection stage)
    opinion_finetune.jsonl  — final opinion generation turns

Each output line is a JSON object in the standard chat format accepted by
Unsloth, Axolotl, LLaMA-Factory, and most HuggingFace fine-tuning tools:

    {
      "conversations": [
        {"role": "system",    "content": "<system prompt>"},
        {"role": "user",      "content": "<user turn>"},
        {"role": "assistant", "content": "<ideal response>"}
      ]
    }

For multi-turn intake conversations, EVERY assistant turn is exported as a
separate training example so the model learns the behaviour at each step of
the conversation, not just the final turn.

Fine-tuning notes (Unsloth + LoRA):
    1. pip install unsloth
    2. Load base model (same one Ollama is running, e.g. llama3 / mistral)
    3. Apply LoRA adapter (r=16, target_modules=["q_proj","v_proj"])
    4. Train on these JSONL files (chat template = ChatML or Alpaca)
    5. Merge adapter weights and export to GGUF for Ollama
    6. ollama create nyaymalaw-ft -f Modelfile

The system prompt baked into each training example is kept SHORT because
after fine-tuning the personality and style are in the weights — you no
longer need 300 lines of instructions in the live prompt.
"""

import json
import os
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent / "examples"
OUTPUT_DIR   = Path(__file__).parent / "finetune_ready"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Short system prompt for fine-tuned model ──────────────────────────────────
# After training, this replaces the current 300-line FACT_COLLECTION_SYSTEM.
# The style, empathy, and prayer-elicitation behaviour live in the weights.

INTAKE_SYSTEM = (
    "You are a senior advocate at Nyaymalaw. "
    "Conduct a warm, human intake conversation with the client. "
    "Acknowledge their situation with genuine empathy before asking anything. "
    "Ask one focused question at a time. "
    "Always elicit what outcome the client is seeking (their prayer or relief). "
    "If they name a specific monetary amount, ask why that figure and what the other party earns. "
    "When you have enough facts, output JSON: "
    '{"action": "complete", "intent": "legal_opinion", '
    '"facts_summary": "<full narrative>", "reply_to_client": "<warm transition>"}. '
    "While gathering facts, output JSON: "
    '{"action": "ask", "reply_to_client": "<acknowledgment + one question + brief reason>"}.'
)

OPINION_SYSTEM = (
    "You are a senior Indian advocate at Nyaymalaw. "
    "Write a structured legal opinion using ONLY the retrieved materials provided. "
    "Format: Facts of the Case → Disputes Identified → Legal Protection (per dispute) → "
    "Reliefs Sought & Assessment (only if prayer mentioned) → Next Steps. "
    "Never cite any section or case not present in the retrieved materials. "
    "Close with honest, grounded encouragement."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping")
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: line {i} in {path.name} is malformed: {e}")
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Written {len(records)} examples → {path}")


# ── Intake converter ──────────────────────────────────────────────────────────

def convert_intake(examples: list[dict]) -> list[dict]:
    """
    For each intake example, produce one training record per assistant turn.
    This teaches the model how to respond at EVERY step of the conversation,
    not just the final one.

    Example with 5 turns (3 user, 2 assistant + 1 final):
        → record 1: history=[user_0] → target=assistant_0
        → record 2: history=[user_0, assistant_0, user_1] → target=assistant_1
        → record 3: history=[user_0 … user_2] → target=assistant_2  (complete)
    """
    records = []
    for ex in examples:
        conversation = ex.get("conversation", [])
        if not conversation:
            continue

        # Build up the context turn by turn; emit a record at each assistant response
        history: list[dict] = []
        for turn in conversation:
            if turn["role"] == "user":
                history.append({"role": "user", "content": turn["content"]})
            elif turn["role"] == "assistant":
                # Emit training record: system + history so far → this assistant reply
                record = {
                    "conversations": (
                        [{"role": "system", "content": INTAKE_SYSTEM}]
                        + history
                        + [{"role": "assistant", "content": turn["content"]}]
                    ),
                    "_meta": {
                        "source_id":   ex["id"],
                        "case_type":   ex["case_type"],
                        "description": ex["description"],
                    }
                }
                records.append(record)
                # Add this assistant turn to history for the next iteration
                history.append({"role": "assistant", "content": turn["content"]})

    return records


# ── Opinion converter ─────────────────────────────────────────────────────────

def convert_opinions(examples: list[dict]) -> list[dict]:
    """
    Each opinion example becomes one training record:
        system + user(facts + placeholder retrieved materials) → ideal opinion

    NOTE: For real fine-tuning, replace the placeholder retrieved-materials
    block with actual retrieval outputs from your vector store so the model
    learns to ground citations in what is passed, not what it memorised.
    """
    records = []
    for ex in examples:
        opinion_text = ex.get("opinion_text", "").strip()
        if not opinion_text:
            continue

        # Reconstruct a plausible user prompt (facts + empty retrieved block)
        user_content = (
            f"CLIENT FACTS:\n{ex['description']}\n\n"
            "RETRIEVED LEGAL MATERIALS GROUPED BY DISPUTE:\n"
            "[Retrieved sections and case laws would be inserted here by the pipeline]\n\n"
            "Please prepare the structured legal opinion."
        )

        record = {
            "conversations": [
                {"role": "system",    "content": OPINION_SYSTEM},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": opinion_text},
            ],
            "_meta": {
                "source_id":   ex["id"],
                "case_type":   ex["case_type"],
                "description": ex["description"],
            }
        }
        records.append(record)

    return records


# ── Stats printer ─────────────────────────────────────────────────────────────

def print_stats(records: list[dict], label: str) -> None:
    if not records:
        return
    total_turns = sum(len(r["conversations"]) for r in records)
    case_types = {}
    for r in records:
        ct = r.get("_meta", {}).get("case_type", "unknown")
        case_types[ct] = case_types.get(ct, 0) + 1
    print(f"\n  {label}: {len(records)} training records, {total_turns} total turns")
    for ct, count in sorted(case_types.items()):
        print(f"    {ct}: {count} records")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n=== Nyaymalaw → Fine-Tune Converter ===\n")

    # Load raw examples
    intake_raw  = load_jsonl(EXAMPLES_DIR / "intake_conversations.jsonl")
    opinion_raw = load_jsonl(EXAMPLES_DIR / "final_opinions.jsonl")
    print(f"Loaded {len(intake_raw)} intake examples, {len(opinion_raw)} opinion examples")

    # Convert
    intake_records  = convert_intake(intake_raw)
    opinion_records = convert_opinions(opinion_raw)

    print_stats(intake_records,  "Intake fine-tune records")
    print_stats(opinion_records, "Opinion fine-tune records")

    # Write output
    write_jsonl(OUTPUT_DIR / "intake_finetune.jsonl",  intake_records)
    write_jsonl(OUTPUT_DIR / "opinion_finetune.jsonl", opinion_records)

    total = len(intake_records) + len(opinion_records)
    print(f"\nTotal fine-tuning records: {total}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("\nNext steps:")
    print("  1. pip install unsloth")
    print("  2. Adjust INTAKE_SYSTEM / OPINION_SYSTEM at top of this file if needed")
    print("  3. Run Unsloth with finetune_ready/intake_finetune.jsonl + opinion_finetune.jsonl")
    print("  4. Export merged model to GGUF → load in Ollama as 'nyaymalaw-ft'")
    print("  5. Gradually shorten live prompts as fine-tuned behaviour is verified\n")


if __name__ == "__main__":
    main()

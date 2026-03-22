"""
Convert Nyaymalaw training examples → Unsloth / QLoRA fine-tuning format
=========================================================================

PRIMARY SOURCE  : nyaymalaw_training_examples.md  (987 case_derived examples)
SECONDARY SOURCE: training/examples/ JSONL files  (legacy rich_training_records)

Usage:
    python training/convert_to_finetune.py

Outputs four files in training/finetune_ready/:
    train.jsonl          — 90% of all examples, shuffled
    val.jsonl            — 10% held-out validation set
    intake_finetune.jsonl   — legacy JSONL-based intake turns (unchanged path)
    opinion_finetune.jsonl  — legacy JSONL-based opinion turns (unchanged path)

Each output line is one JSON object in OpenAI-compatible chat format:

    {
      "messages": [
        {"role": "system",    "content": "<system prompt>"},
        {"role": "user",      "content": "<client scenario>"},
        {"role": "assistant", "content": "<full advocate response>"}
      ]
    }

Citation masking policy
-----------------------
Case names INSIDE the "Grounded Advice Layer" section are replaced with
[RELEVANT_PRECEDENT]. This prevents the model from memorising specific
citations that should come from retrieval at inference time.
The retrieval-trigger FORMAT (Retrieve ... with ... principles) is preserved
so the model learns the pattern without the hardcoded name.

Example citations outside the Grounded Advice Layer (e.g. in Weakness
Stress-Test prose) are left untouched — they appear rarely and provide
useful factual grounding for the analysis sections.

Token length notes (for RTX 4060 8GB, max_seq_length=3072)
----------------------------------------------------------
A warning is printed for any example exceeding ~3000 estimated tokens.
These examples are NOT dropped — they are included but you should monitor
whether they get truncated by the trainer. Consider running with
max_seq_length=4096 if more than 5% of examples trigger the warning.
"""

import json
import os
import random
import re
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).parent.parent
EXAMPLES_DIR = Path(__file__).parent / "examples"
OUTPUT_DIR   = Path(__file__).parent / "finetune_ready"
OUTPUT_DIR.mkdir(exist_ok=True)

MD_SOURCE  = REPO_ROOT / "nyaymalaw_training_examples.md"
RICH_FILE  = EXAMPLES_DIR / "rich_cases" / "rich_training_records.jsonl"

RANDOM_SEED   = 42
TRAIN_RATIO   = 0.90
WARN_TOKENS   = 3000   # rough char/4 estimate; actual BPE tokens will differ

# ── System prompts ─────────────────────────────────────────────────────────────
# Kept SHORT — after fine-tuning, style and behaviour live in the weights.
# The live runtime prompt can be shortened once fine-tuned behaviour is verified.

INTAKE_SYSTEM = (
    "You are a senior advocate at Nyaymalaw, an Indian legal platform. "
    "When a client presents their situation, conduct a structured intake: "
    "acknowledge their problem with genuine empathy, ask focused one-at-a-time "
    "questions to surface the legal issues, assess urgency and client type, "
    "identify what relief they are seeking, and produce a decision-state snapshot. "
    "Do NOT cite section numbers, act names, or case names from memory during intake. "
    "Grounded legal citations are handled exclusively by the retrieval layer."
)

OPINION_SYSTEM = (
    "You are a senior Indian advocate at Nyaymalaw. "
    "Write a structured legal opinion using ONLY the retrieved materials provided. "
    "Format: Facts → Disputes Identified → Legal Protection (per dispute) → "
    "Reliefs & Assessment → Next Steps. "
    "Never cite any section or case not present in the retrieved materials. "
    "Close with honest, grounded encouragement."
)

# System prompt for Junior Advocate mentoring examples.
# These teach decision-value question selection and contrastive reasoning.
JA_SYSTEM = (
    "You are a senior advocate at Nyaymalaw mentoring a junior colleague. "
    "When the junior presents a legal situation or proposes an approach, "
    "guide them to the highest-value question or action for that stage. "
    "Explain which decision dimension the question targets, why it is the "
    "right priority, and what a weaker alternative would miss. "
    "Do NOT cite section numbers or case names from memory — "
    "ground all citation-level advice in retrieved materials only."
)

# ── Citation masking ───────────────────────────────────────────────────────────

# Matches italic case names: *Some Case v. Someone* or *Case Name*
# Applied ONLY to the Grounded Advice Layer block.
_ITALIC_CASE_RE = re.compile(r'\*([A-Z][^*\n]{2,80})\*')

def _mask_citations_in_grounded_layer(text: str) -> str:
    """
    Replace specific case names with [RELEVANT_PRECEDENT] inside the
    Grounded Advice Layer section only.

    The retrieval-trigger sentence structure is preserved:
      Before: "Retrieve *Md. Allauddin Khan* with current civil-pendency principles."
      After:  "Retrieve *[RELEVANT_PRECEDENT]* with current civil-pendency principles."
    """
    # Split into sections at ### headers
    sections = re.split(r'(\n### )', text)
    result = []
    inside_grounded = False

    for part in sections:
        if part == '\n### ':
            result.append(part)
            inside_grounded = False  # reset; next part will reveal the header name
        elif inside_grounded:
            # Mask italic spans that look like case names in this section
            masked = _ITALIC_CASE_RE.sub(r'*[RELEVANT_PRECEDENT]*', part)
            result.append(masked)
        else:
            # Check if this part starts with "Grounded Advice Layer"
            if part.lstrip().startswith('Grounded Advice Layer'):
                inside_grounded = True
                masked = _ITALIC_CASE_RE.sub(r'*[RELEVANT_PRECEDENT]*', part)
                result.append(masked)
            else:
                result.append(part)

    return ''.join(result)


# ── Markdown parser ────────────────────────────────────────────────────────────

# Sections that go into the assistant response (in order)
_ASSISTANT_SECTIONS = [
    "Enriched Intake Conversation",
    "Decision-State Snapshot",
    "Weakness Stress-Test",
    "Intake-Layer Action Plan",
    "Case-Strengthening Guidance",
    "Potential Judgment",          # matches "Potential Judgment / Outcome"
    "Grounded Advice Layer",
]

def _extract_section(body: str, section_name: str) -> str:
    """
    Extract content between a ### section_name header and the next ### header.
    Returns empty string if not found.
    """
    pattern = rf'### {re.escape(section_name)}(.*?)(?=\n###|\Z)'
    m = re.search(pattern, body, re.DOTALL | re.IGNORECASE)
    if not m:
        # Try partial match (e.g. "Potential Judgment" for "Potential Judgment / Outcome")
        pattern2 = rf'### {re.escape(section_name)}[^\n]*\n(.*?)(?=\n###|\Z)'
        m = re.search(pattern2, body, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ''


def _build_user_turn(body: str, header: str) -> str:
    """
    User turn = the Client Scenario section ONLY.
    The case header (which contains the real case name) is intentionally
    excluded — the model should not learn to associate specific case names
    with intake scenarios. The _meta field preserves traceability.
    """
    scenario = _extract_section(body, "Client Scenario")
    return scenario if scenario else ''


def _build_assistant_turn(body: str) -> str:
    """
    Assistant turn = concatenation of all relevant sections with their headers,
    with citation masking applied to the Grounded Advice Layer.
    """
    parts = []
    for section in _ASSISTANT_SECTIONS:
        content = _extract_section(body, section)
        if not content:
            continue
        # Find the actual header name as it appears (may have " / Outcome" suffix)
        header_match = re.search(
            rf'### ({re.escape(section)}[^\n]*)', body, re.IGNORECASE)
        actual_header = header_match.group(1) if header_match else section
        parts.append(f"### {actual_header}\n\n{content}")

    full_response = "\n\n".join(parts)

    # Apply citation masking to the entire response (function targets only the
    # Grounded Advice Layer section internally)
    return _mask_citations_in_grounded_layer(full_response)


def parse_markdown_examples(md_path: Path) -> list[dict]:
    """
    Parse nyaymalaw_training_examples.md and return a list of dicts with keys:
        id, domain, client_type, matter_shape, header, user_turn, assistant_turn
    """
    text = md_path.read_text(encoding='utf-8')

    # Split on case_derived headers — capture header and body
    # Pattern: ### case_derived_NNN: Some Title
    chunks = re.split(r'\n(### case_derived_\d+:[^\n]+)\n', text)

    examples = []
    skipped  = 0

    for i in range(1, len(chunks), 2):
        header = chunks[i]           # e.g. "### case_derived_956: ..."
        body   = chunks[i + 1]       # everything until next header

        # Extract ID
        id_match = re.search(r'case_derived_(\d+)', header)
        if not id_match:
            skipped += 1
            continue
        case_id = int(id_match.group(1))

        # Extract metadata from the table
        def _meta(field: str) -> str:
            m = re.search(
                rf'\|\s*\*\*{re.escape(field)}\*\*\s*\|\s*([^|\n]+)\s*\|',
                body)
            return m.group(1).strip() if m else ''

        domain       = _meta('Domain')
        client_type  = _meta('Client Type')
        matter_shape = _meta('Matter Shape')

        user_turn      = _build_user_turn(body, header.lstrip('# '))
        assistant_turn = _build_assistant_turn(body)

        if not user_turn or not assistant_turn:
            skipped += 1
            continue

        examples.append({
            'id':            case_id,
            'domain':        domain,
            'client_type':   client_type,
            'matter_shape':  matter_shape,
            'header':        header.lstrip('# '),
            'user_turn':     user_turn,
            'assistant_turn': assistant_turn,
        })

    print(f"  Parsed {len(examples)} examples from markdown "
          f"({skipped} skipped — missing required sections)")
    return examples


# ── Convert markdown examples to chat format ──────────────────────────────────

def convert_markdown_to_chat(examples: list[dict]) -> list[dict]:
    """
    Convert parsed markdown examples to the messages chat format.
    Adds token-length warning for long examples.
    """
    records    = []
    long_count = 0

    for ex in examples:
        assistant_content = ex['assistant_turn']
        user_content      = ex['user_turn']

        # Rough token estimate: chars / 3.5 (conservative for legal English)
        est_tokens = (len(INTAKE_SYSTEM) + len(user_content) +
                      len(assistant_content)) / 3.5

        if est_tokens > WARN_TOKENS:
            long_count += 1

        record = {
            "messages": [
                {"role": "system",    "content": INTAKE_SYSTEM},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "_meta": {
                "source_id":    f"case_derived_{ex['id']}",
                "domain":       ex['domain'],
                "client_type":  ex['client_type'],
                "matter_shape": ex['matter_shape'],
                "est_tokens":   round(est_tokens),
            }
        }
        records.append(record)

    if long_count:
        pct = 100 * long_count / len(records)
        print(f"  WARNING: {long_count} examples ({pct:.1f}%) estimated > "
              f"{WARN_TOKENS} tokens. Consider max_seq_length=4096 if > 5%.")

    return records


# ── Train / val split ─────────────────────────────────────────────────────────

def split_train_val(
    records: list[dict],
    train_ratio: float = TRAIN_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict]]:
    """
    Stratified-ish split: shuffle then split at train_ratio boundary.
    Stratification is on matter_shape so each split sees all three shapes.
    """
    rng = random.Random(seed)

    # Group by matter_shape
    groups: dict[str, list[dict]] = {}
    for rec in records:
        shape = rec.get('_meta', {}).get('matter_shape', 'Unknown')
        groups.setdefault(shape, []).append(rec)

    train, val = [], []
    for shape, group in groups.items():
        rng.shuffle(group)
        cut = max(1, round(len(group) * train_ratio))
        train.extend(group[:cut])
        val.extend(group[cut:])

    # Final shuffle so shape order is mixed
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ── JSONL helpers (legacy pipeline, unchanged) ────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  INFO: {path} not found — skipping legacy source")
        return []
    out = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: line {i} in {path.name} malformed: {e}")
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"  Written {len(records):>5} examples → {path.relative_to(REPO_ROOT)}")


# ── Legacy JSONL converters (kept for backward compatibility) ─────────────────

def _conv(ex: dict) -> list[dict]:
    return (ex.get('conversation')
            or ex.get('intake_conversation', {}).get('conversation')
            or [])

def _opinion(ex: dict) -> str:
    return (ex.get('opinion_text')
            or ex.get('final_opinion', {}).get('opinion_text') or '').strip()

def _desc(ex: dict) -> str:
    return (ex.get('description')
            or ex.get('client_scenario', {}).get('presenting_problem')
            or ex.get('case_name') or '')

def _domain(ex: dict) -> str:
    return ex.get('case_type') or ex.get('legal_domain') or 'unknown'


def convert_legacy_intake(examples: list[dict]) -> list[dict]:
    records = []
    for ex in examples:
        conversation = _conv(ex)
        if not conversation:
            continue
        history: list[dict] = []
        for turn in conversation:
            if turn['role'] == 'user':
                history.append({'role': 'user', 'content': turn['content']})
            elif turn['role'] == 'assistant':
                record = {
                    'messages': (
                        [{'role': 'system', 'content': INTAKE_SYSTEM}]
                        + history
                        + [{'role': 'assistant', 'content': turn['content']}]
                    ),
                    '_meta': {
                        'source_id':   ex.get('id', ''),
                        'domain':      _domain(ex),
                        'description': _desc(ex),
                    }
                }
                records.append(record)
                history.append({'role': 'assistant', 'content': turn['content']})
    return records


def convert_legacy_opinions(examples: list[dict]) -> list[dict]:
    records = []
    for ex in examples:
        opinion_text = _opinion(ex)
        if not opinion_text:
            continue
        user_content = (
            f"CLIENT FACTS:\n{_desc(ex)}\n\n"
            "RETRIEVED LEGAL MATERIALS GROUPED BY DISPUTE:\n"
            "[Retrieved sections and case laws would be inserted here by the pipeline]\n\n"
            "Please prepare the structured legal opinion."
        )
        record = {
            'messages': [
                {'role': 'system',    'content': OPINION_SYSTEM},
                {'role': 'user',      'content': user_content},
                {'role': 'assistant', 'content': opinion_text},
            ],
            '_meta': {
                'source_id':   ex.get('id', ''),
                'domain':      _domain(ex),
                'description': _desc(ex),
            }
        }
        records.append(record)
    return records


def merge_by_id(*groups: list[dict]) -> list[dict]:
    out, seen = [], set()
    for group in groups:
        for ex in group:
            ex_id = ex.get('id')
            if not ex_id or ex_id in seen:
                continue
            seen.add(ex_id)
            out.append(ex)
    return out


# ── Junior Advocate dialog parser ────────────────────────────────────────────

def _parse_ja_dialog_turns(body: str) -> list[dict]:
    """
    Parse a pure-dialog Junior Advocate example into alternating role/content pairs.

    Input format (repeating):
        **Junior Advocate:** <question text>

        **Senior Advocate:** <answer text>

        *Decision dimension targeted:* ...
        *Why this question wins:* ...
        *Plausible bad alternative:* ...
        *Why bad alternative loses:* ...

        **Junior Advocate:** <follow-up>   ← next user turn starts here

    The annotation block (*Decision dimension*, *Why wins*, etc.) is attached
    to the PRECEDING Senior Advocate turn so the model learns to produce
    decision-dimension and contrastive annotations as part of its response.

    Returns a list of {"role": "user"|"assistant", "content": str}.
    """
    # Split on speaker markers; keep the speaker name in the result
    parts = re.split(r'\*\*(Junior Advocate|Senior Advocate):\*\*\s*', body)
    # parts[0] = preamble (metadata table, use-case line) — discard
    # parts[1], parts[2] = 'Junior Advocate', <text>
    # parts[3], parts[4] = 'Senior Advocate', <text + annotations>
    # …

    turns: list[dict] = []
    i = 1
    while i < len(parts) - 1:
        speaker = parts[i].strip()          # 'Junior Advocate' or 'Senior Advocate'
        content = parts[i + 1].strip()      # everything until next speaker marker
        i += 2

        if not content:
            continue

        role = 'user' if speaker == 'Junior Advocate' else 'assistant'
        turns.append({'role': role, 'content': content})

    return turns


def parse_ja_examples(md_path: Path) -> list[dict]:
    """
    Find and parse all pure-dialog Junior Advocate examples (no standard section
    headers) from the markdown file.

    Returns a list of dicts with keys: id, domain, client_type, turns
    where turns is a list of {"role", "content"} pairs.
    """
    text   = md_path.read_text(encoding='utf-8')
    chunks = re.split(r'\n(### case_derived_\d+:[^\n]+)\n', text)

    examples = []
    skipped  = 0

    for i in range(1, len(chunks), 2):
        header = chunks[i]
        body   = chunks[i + 1]

        has_enriched  = bool(re.search(r'### Enriched Intake Conversation', body))
        has_ja_dialog = bool(re.search(r'\*\*Junior Advocate:\*\*', body))

        # Only process examples that have JA dialog but no standard section headers
        if has_enriched or not has_ja_dialog:
            continue

        id_match = re.search(r'case_derived_(\d+)', header)
        if not id_match:
            skipped += 1
            continue

        def _meta(field: str) -> str:
            m = re.search(
                rf'\|\s*\*\*{re.escape(field)}\*\*\s*\|\s*([^|\n]+)\s*\|', body)
            return m.group(1).strip() if m else ''

        turns = _parse_ja_dialog_turns(body)
        if len(turns) < 2:          # need at least one JA + one SA turn
            skipped += 1
            continue

        examples.append({
            'id':          int(id_match.group(1)),
            'domain':      _meta('Domain'),
            'client_type': 'Junior Advocate',
            'turns':       turns,
        })

    print(f"  Parsed {len(examples)} Junior Advocate examples "
          f"({skipped} skipped)")
    return examples


def convert_ja_to_chat(examples: list[dict]) -> list[dict]:
    """
    Convert JA dialog examples to multi-turn chat records.

    Each example becomes ONE record with alternating user/assistant turns.
    This preserves the full mentoring flow (including all contrastive
    annotations) as a coherent training sequence.

    Citation masking is applied to the full assistant content to catch any
    case names that appear inside *italic* spans within annotations.
    """
    records = []

    for ex in examples:
        messages = [{'role': 'system', 'content': JA_SYSTEM}]

        for turn in ex['turns']:
            content = turn['content']
            if turn['role'] == 'assistant':
                # Apply masking to italic spans in SA turns (annotations may
                # contain case names in the form *Case v. Respondent*)
                content = _ITALIC_CASE_RE.sub(r'*[RELEVANT_PRECEDENT]*', content)
            messages.append({'role': turn['role'], 'content': content})

        # Rough token estimate
        total_chars = sum(len(m['content']) for m in messages)
        est_tokens  = total_chars / 3.5

        records.append({
            'messages': messages,
            '_meta': {
                'source_id':    f"case_derived_{ex['id']}",
                'domain':       ex['domain'],
                'client_type':  'Junior Advocate',
                'matter_shape': 'JA-dialog',
                'est_tokens':   round(est_tokens),
            }
        })

    return records


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats(records: list[dict], label: str) -> None:
    if not records:
        return
    shapes: dict[str, int] = {}
    domains: dict[str, int] = {}
    token_sum = 0
    for r in records:
        meta = r.get('_meta', {})
        s = meta.get('matter_shape', meta.get('domain', 'unknown'))
        shapes[s] = shapes.get(s, 0) + 1
        d = meta.get('domain', 'unknown')
        domains[d] = domains.get(d, 0) + 1
        token_sum += meta.get('est_tokens', 0)

    avg_tokens = token_sum / len(records) if records else 0
    print(f"\n  {label}: {len(records)} records  "
          f"(avg ~{avg_tokens:.0f} est. tokens)")

    if shapes:
        print("  Matter shape breakdown:")
        for shape, count in sorted(shapes.items(), key=lambda x: -x[1]):
            bar = '█' * (count // 10)
            print(f"    {shape:<20} {count:>4}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print('\n=== Nyaymalaw → Fine-Tune Converter (v2) ===\n')

    # ── 1. Parse markdown source (primary, 987 examples) ──────────────────────
    print(f"Reading markdown source: {MD_SOURCE.name}")
    md_examples  = parse_markdown_examples(MD_SOURCE)
    md_records   = convert_markdown_to_chat(md_examples)

    # ── 2. Parse Junior Advocate dialog examples (previously skipped 30) ─────
    print("\nParsing Junior Advocate dialog examples...")
    ja_examples = parse_ja_examples(MD_SOURCE)
    ja_records  = convert_ja_to_chat(ja_examples)
    print(f"  Converted {len(ja_records)} JA multi-turn records")

    # ── 3. Combine and train / val split ─────────────────────────────────────
    all_records = md_records + ja_records
    train_records, val_records = split_train_val(all_records)
    print(f"\n  Train split: {len(train_records)} examples")
    print(f"  Val   split: {len(val_records)} examples")

    write_jsonl(OUTPUT_DIR / 'train.jsonl', train_records)
    write_jsonl(OUTPUT_DIR / 'val.jsonl',   val_records)

    print_stats(train_records, 'Train set')
    print_stats(val_records,   'Validation set')

    # ── 3. Legacy JSONL pipeline (secondary, preserving existing outputs) ─────
    print('\nProcessing legacy JSONL sources...')
    rich_raw  = load_jsonl(RICH_FILE)
    leg_in    = load_jsonl(EXAMPLES_DIR / 'intake_conversations.jsonl')
    leg_op    = load_jsonl(EXAMPLES_DIR / 'final_opinions.jsonl')
    merged    = merge_by_id(rich_raw, leg_in, leg_op)
    print(f"  {len(merged)} unique legacy records")

    legacy_intake   = convert_legacy_intake(merged)
    legacy_opinions = convert_legacy_opinions(merged)

    write_jsonl(OUTPUT_DIR / 'intake_finetune.jsonl',  legacy_intake)
    write_jsonl(OUTPUT_DIR / 'opinion_finetune.jsonl', legacy_opinions)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    total_primary = len(train_records) + len(val_records)
    total_legacy  = len(legacy_intake) + len(legacy_opinions)
    print(f'\n{"="*55}')
    print(f'  Enriched intake examples    : {len(md_records)}')
    print(f'  Junior Advocate examples    : {len(ja_records)}')
    print(f'  Total primary examples      : {total_primary}')
    print(f'    → train.jsonl             : {len(train_records)}')
    print(f'    → val.jsonl               : {len(val_records)}')
    print(f'  Legacy JSONL examples       : {total_legacy}')
    print(f'{"="*55}')
    print(f'\nOutput directory: {OUTPUT_DIR.resolve()}')
    print('\nNext steps:')
    print('  1. python training/convert_to_finetune.py   ← you are here')
    print('  2. Review a sample: head -1 training/finetune_ready/train.jsonl | python -m json.tool')
    print('  3. Run Unsloth training with train.jsonl + val.jsonl')
    print('  4. Monitor: if > 5% of examples hit token warning, set max_seq_length=4096')
    print('  5. After training, export merged model → GGUF → Ollama\n')


if __name__ == '__main__':
    main()

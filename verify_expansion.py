"""
Verify expand_legal_query() works correctly after the Chat Completions fix.
Run from the NM_API-1.0 directory:
    python /sessions/dreamy-sweet-hawking/verify_expansion.py
"""
import sys, json, os, textwrap

sys.path.insert(0, "/sessions/dreamy-sweet-hawking/mnt/NM_API-1.0")

# Load .env so API keys are available
try:
    from dotenv import load_dotenv
    load_dotenv("/sessions/dreamy-sweet-hawking/mnt/NM_API-1.0/.env")
except Exception:
    pass

from services.response_generator_v2 import expand_legal_query

# ---------------------------------------------------------------------------
# Three test cases that exercise different legal domains
# ---------------------------------------------------------------------------
CASES = [
    {
        "label": "Government land encroachment",
        "facts": (
            "My neighbour has encroached on government-owned land adjacent to my property "
            "in Pune, Maharashtra. He has constructed a boundary wall on it. I complained "
            "to the local collector's office but no action was taken for 6 months. "
            "I want the encroachment removed and damages."
        ),
    },
    {
        "label": "Domestic violence + maintenance",
        "facts": (
            "My husband has been physically abusing me and my two minor children for the past "
            "two years in Delhi. He threw us out of the matrimonial home last month and has "
            "stopped paying any maintenance. I want protection, residence rights, and monthly "
            "maintenance for myself and the children."
        ),
    },
    {
        "label": "Cheque bounce + salary dues",
        "facts": (
            "My employer issued a cheque of Rs 1,80,000 towards three months unpaid salary. "
            "The cheque bounced due to insufficient funds. I sent a legal notice but no payment "
            "came within 15 days. The company is registered in Bangalore. I want to recover "
            "the money and also claim compensation for the dishonour."
        ),
    },
]

SEP = "─" * 72

for case in CASES:
    debug: dict = {}
    result = expand_legal_query(
        facts=case["facts"],
        intent=None,
        expansion_debug=debug,
        model_override=None,  # use default (OpenAI fast model)
    )

    print(f"\n{SEP}")
    print(f"CASE: {case['label']}")
    print(SEP)
    print("INPUT FACTS (truncated):")
    print(textwrap.fill(case["facts"], width=70, initial_indent="  ", subsequent_indent="  "))
    print()

    raw = debug.get("model_raw_response", "")
    if raw:
        print("MODEL RAW RESPONSE:")
        print(textwrap.indent(raw[:800], "  "))
        print()

    issues = debug.get("issues_from_model", [])
    if issues:
        print("PARSED ISSUES + QUERIES:")
        for iss in issues:
            print(f"  [{iss.get('issue_label', '(unlabelled)')}]")
            for q in iss.get("queries", []):
                print(f"    • {q}")
        print()
    else:
        print("  (no structured issues parsed — fallback used)")
        print()

    added = debug.get("queries_added_after_model", [])
    if added:
        print(f"FALLBACK QUERIES ADDED (model returned nothing usable):")
        for q in added:
            print(f"    ⚠  {q}")
        print()

    print(f"FINAL QUERIES GOING TO RETRIEVAL ({len(result)} total):")
    for i, q in enumerate(result, 1):
        print(f"  {i:>2}. {q}")

print(f"\n{SEP}")
print("DONE")

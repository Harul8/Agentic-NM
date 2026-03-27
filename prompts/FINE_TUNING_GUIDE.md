# Prompt and Few-Shot Guide

This folder controls Nyaymalaw's runtime behavior when using the base model.
The app now relies on two things working together:

1. Light system prompts in `prompts/advocate_prompts.py`
2. Small retrieved few-shot examples from `training/few_shot_retriever.py`

The idea is simple: keep the prompts principle-driven, and let examples teach tone,
flow, and output shape.

## Runtime layers

### 1. Intake state extraction
What it does:
- reads the current chat
- decides whether this is greeting, generic chat, search, lookup, or legal intake
- returns compact state only

Where to tune:
- `INTAKE_STATE_UPDATE_SYSTEM`
- few-shot source: `get_intake_state_example_pack()`

### 2. Next intake move
What it does:
- decides whether to ask the next question cluster or complete intake
- keeps the conversation warm, purposeful, and non-repetitive

Where to tune:
- `NEXT_QUESTION_FROM_STATE_SYSTEM`
- few-shot source: `get_intake_reply_example_pack()`

### 3. Bare-act follow-up
What it does:
- briefly explains the most relevant retrieved sections
- asks for more facts only if one compact gap still matters materially

Where to tune:
- `BARE_ACT_EXPLAIN_AND_FOLLOWUP_PROMPT`

### 4. Final grounded opinion
What it does:
- writes the final opinion from retrieved materials only
- uses short fact framing, grounded analysis, and practical guidance

Where to tune:
- `STRUCTURED_FINAL_OPINION_BY_DISPUTE_PROMPT`
- `STRUCTURED_FINAL_OPINION_PROMPT`
- `RELEVANCE_EXPLANATION_SYSTEM`
- few-shot source: `get_opinion_example()`

## Few-shot data

Default runtime few-shot examples are loaded from:
- `training/runtime_fewshot/runtime_examples.jsonl`

You can override the source files with:
- `NYAYMALAW_FEWSHOT_FILES`

Use this when you want runtime prompting to read a different curated example set without
changing code. The default runtime path now avoids the full training corpus.

## Runtime flags

- `ENABLE_INTAKE_FEWSHOT=1` enables intake examples in `fact_collector.py`
- `ENABLE_RUNTIME_FEWSHOT=1` enables final-opinion examples in `response_generator_v2.py`

## Design rule

Do not hard-code case-specific scripts into the system prompts.
If you want the model to learn a style, a completion pattern, or a better handoff,
put that behavior into the few-shot examples instead of bloating the prompt.

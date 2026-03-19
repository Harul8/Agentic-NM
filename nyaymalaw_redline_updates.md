# Nyaymalaw - Redline Update Notes v1.1

This file records the precise wording changes made after review of:
- schema compactness in runtime use
- stop policy enforceability
- boundary between intake diagnosis and retrieval-grounded legal advice

---

## 1. Prompt Redlines

### Added
- New section: `Layer Boundary (Critical)`
  - Added explicit distinction between:
    - `Intake / Diagnostic Layer`
    - `Final Legal Advice / Grounded Analysis Layer`
  - Added rule that intake may frame issues and strategy in abstract terms but must not rely on memory for section numbers, case names, or conclusive propositions of law.
  - Added rule that citation-backed legal advice must come only from retrieval.

### Changed
- `Step 7 - Identify Legal Issues`
  - Old emphasis: issue identification only
  - New emphasis: issue identification plus explicit instruction not to jump to unsupported citations from memory at intake stage

- `Stop / Exit Policy`
  - Old wording:
    - stop when objective known, urgency known, and next step known
  - New wording:
    - stop only when those three are known **and** there is no unresolved fact likely to materially change:
      - forum / jurisdiction
      - maintainability
      - limitation
      - evidence viability
      - immediate procedural safety
      - defence exposure
  - Added `Counter-test before stopping`

- `Output Format`
  - Lay client:
    - changed `No section citations unless essential`
    - to `No section citations unless they come from approved retrieval and are essential`
  - Junior advocate:
    - added `At intake stage, teach the diagnostic before giving citation-backed law`

- `What You Are Not`
  - Added:
    - `You are not here to give citation-backed final legal advice before retrieval.`

---

## 2. Schema Redlines

### Added
- `_runtime_guidance`
  - clarifies that the schema is a decision-state schema, not a chain-of-thought dump

- `_core_runtime_fields`
  - added explicit compact runtime core to prevent mechanical overpopulation

- `_optional_refinement_fields`
  - added explicit optional layer so richness stays available without being mandatory every turn

- `_layer_boundary`
  - added:
    - `intake_layer_rule`
    - `grounded_advice_rule`

- `output._output_layer_note`
  - added explicit instruction that citation-backed final advice belongs to the grounded advice layer

### Changed
- `decision_state._enough_to_advise_note`
  - Old wording:
    - objective known, urgency assessed, next step clear
  - New wording:
    - objective known, urgency assessed, next step clear,
    - and no unresolved fact likely to materially change forum, maintainability, limitation, evidence viability, defence exposure, or immediate procedural safety

---

## 3. Eval Checklist Redlines

### Changed
- Version updated from `v1.0` to `v1.1`
- Test count updated from `15` to `18`
- Hard block count updated from `7` to `9`

### Added Hard Blocks
- `Test 8: Stop Policy Safety`
  - distinguishes premature stopping from healthy stopping

- `Test 9: Intake vs Grounded Advice Boundary`
  - checks that intake does not invent citation-backed law before retrieval

### Renumbered
- Former `Test 8: Stop Policy Compliance` became `Test 10`
- Former tests `9-15` shifted to `11-17`

### Added Quality Test
- `Test 18: Runtime Schema Discipline`
  - checks compact schema use in runtime

### Changed Deployment Threshold
- Old:
  - all 7 Tier 1 pass
  - at least 6 of 8 Tier 2 pass
- New:
  - all 9 Tier 1 pass
  - at least 6 of 9 Tier 2 pass

---

## 4. Example Redlines

### Added
- New boundary note near top:
  - intake examples teach diagnostic behavior
  - final advice examples belong to the grounded legal-advice layer

### Changed
- Example 3 client opening:
  - removed specific memory citation `Section 406 IPC`
  - replaced with generic `criminal breach of trust`

- Example 3 ideal follow-up:
  - changed threshold wording from `Section 406`
  - to `criminal breach of trust`

- Example 3 legal issues snapshot:
  - changed from `Criminal breach of trust (Section 406 IPC / BNS equivalent)`
  - to `Criminal breach of trust`

- Example 3 final advice preface:
  - added explicit retrieval-grounded qualifier

- Final advice headings in examples:
  - updated to clarify they belong to the grounded legal-advice layer, not pure intake

---

## Summary

These updates do three things:

1. keep runtime schema use compact and decision-oriented
2. make the stop policy operational rather than aspirational
3. preserve a hard boundary between intake diagnosis and retrieval-grounded legal advice

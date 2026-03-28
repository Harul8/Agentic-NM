# Motor Accident Live Forensics (Post-fix v3)

Date: 2026-03-27
Workspace: C:\Users\rahul\Nyaymalaw-5.0
Server: http://127.0.0.1:8016
Trace: [motor_accident_live_forensics_postfix_v3_trace.json](C:/Users/rahul/Nyaymalaw-5.0/docs/motor_accident_live_forensics_postfix_v3_trace.json)

## Scenario
Initial facts:
`I was injured in a road accident six days ago when a car hit my motorcycle near Mysuru. I want compensation for my medical expenses, lost income, and bike damage.`

Follow-up facts:
1. FIR number, hospital discharge summary, X-ray report, repair estimate, vehicle registration, rash/negligent police record.
2. Driver had insurance, missed six days of work, employer can confirm salary/absence, pharmacy bills and bank statements.

## What Changed Before This Run
- Intake state extraction no longer escalates from fast model to the slower default model.
- Intake next-question repair also stays on the fast model instead of switching to the slower default model.
- Interactive query expansion now uses the fast model and the compressed focus-fact seed.
- Case-law ranking now gets a shared post-processing alignment pass against shortlisted statutes.
- Startup now performs a small synchronous critical warmup before the heavier background warmup.

## Latest Live Results
1. Turn 1 intake question
   - Latency: 15.018s
   - Model used: Llama 3.2 3B
   - Output quality: empathetic and case-specific
2. Turn 2 follow-up
   - Latency: 2.301s
   - Model used: Llama 3.2 3B
   - Output quality: correctly moved to analysis-ready prompt
3. Turn 3 one-last-fact to final opinion
   - Latency: 28.637s
   - First streamed token: 1.373s
   - Model used: Llama 3.2 3B on the API side for the final result envelope; opinion generation stayed in the fast local interactive path and streamed almost immediately
   - Output quality: materially better than the earlier failure mode; it stayed in the road-accident / compensation lane and cited relevant accident-compensation precedents

## Comparison Against The Earlier Broken Run
- First intake turn improved from about 37.7s to about 15.0s.
- The worst intake follow-up improved from about 110.1s to about 2.3s.
- Final analysis improved from about 424.9s total with first token at about 313.0s to about 28.6s total with first token at about 1.37s.
- The huge pre-retrieval stall caused by long-budget query expansion is gone.

## What The Model Did Well
- Stayed fully model-driven in intake rather than falling back into deterministic question templates.
- Recognized that enough facts existed after the second user follow-up and offered the analysis-ready transition.
- Accepted the user's last fact and moved into opinion generation with streaming feedback.
- Produced a better-grounded accident-compensation opinion than the earlier run that collapsed into irrelevant or empty retrieval.

## Remaining Gaps
- The very first intake turn is still above the single-digit target at about 15s on a cold-to-warm live run.
- The fast local bare-act path is still weak for this motor-accident fact pattern; the final opinion leaned more on precedents than statutes.
- The local case-law rescue path still logs missing `caselaws_v2_chunks.json`, so some rescue quality depends on legacy fallback.
- If a user says `proceed` after analysis is already complete, the system may re-enter analysis unnecessarily. This did not affect the intended three-turn flow, but it showed up when I forced an extra fourth turn in testing.

## Practical Read
The two main fixes worked:
- intake is much less likely to fall into slow-model latency spikes
- response generation no longer spends minutes stuck before retrieval begins

The biggest remaining production lever is now first-turn intake latency, followed by strengthening the local statutory retrieval path for mainstream compensation matters.

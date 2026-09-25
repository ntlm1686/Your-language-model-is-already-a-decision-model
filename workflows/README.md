# Jev official workflow eval feasibility (2026-09-25)

Primary sources: [TypeSafe launch blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
and [official workflow eval viewer](https://evals.typesafe.ai/).

| Workflow | Site-reported evaluation cases | Public showcase cases | Jev-path question instances in public cases |
| --- | ---: | ---: | ---: |
| Security incidents | 240 | 5 | 26 |
| Agent trace observability | 117 | 5 | 52 |
| Invoice processing | 150 | 5 | 232 |
| Customer service | 204 | 5 | 98 |
| **Total** | **711** | **20** | **408** |

The official site's dynamic `*-cases.js` viewer assets expose the 20 selected
showcase cases, their state documents, question definitions, saved model answers,
and some reference answers. `download_examples.py` saves exact current assets
and SHA-256 hashes in `manifest.json`. `build_replay.py` flattens the 408
questions on Jev's saved paths to `public_question_replay.jsonl`. Of these,
354 have reference answers in the viewer asset. The remaining 54 should not
be scored against a fabricated reference.
All 408 replay records were successfully converted by the local CLM
`clm.schema.build_pairs` adapter (264 Noul, 117 Choice, 27 Score), so the
published examples are directly usable for a question-level CLM smoke test.

The showcase cases were selected to display model disagreements and agreements.
They are not a random sample. The full 711 case inputs, executable policy
harness, and reference generation procedure are not published in the viewer.
Thus the site's end-to-end accuracy, latency, and cost chart cannot be exactly
reproduced from the public assets. A question-level replay can measure local
CLM/Qwen agreement with the saved reference and record latency/cost, but it
is a diagnostic on selected examples, not a leaderboard reproduction.

Jev is accessed via TypeSafe's API; no `TYPESAFE_API_KEY` or `JEV_API_KEY` is
configured in this environment. Consequently a fresh Jev run requires API
access. The public assets include saved Jev outputs, which can be compared
without making new Jev calls, but their old latency/cost cannot substitute
for a matched current measurement.

For a benchmark directly relevant to agent routing, build a new test set with
independent outcomes (for example, DeepSWE verifier success) and score exactly
the same state/action choices with Jev, CLM, and a Qwen baseline. Report
selection success, latency, input tokens, and calibration separately. The
official Agent Trace Observability workflow checks completed support traces
for attention needed; it is related but not the same as choosing among four
coding trajectories.

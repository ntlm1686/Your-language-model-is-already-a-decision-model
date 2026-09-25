# PhishNChips email decision comparison

The 2,000-email run uses the verified [jev-phishing-bench](https://github.com/anisselbd/jev-phishing-bench) preparation of PhishNChips v5.2. Every method sees the same six email fields and the same binary verdict descriptions. See the [protocol](benchmark_protocol.md#phishnchips-email-decision) for commands, source commits, and prompt details.

| Method | Correct / 2,000 | Phishing recall | False positive rate |
| --- | ---: | ---: | ---: |
| URL host rule | **1,833 (91.65%)** | 83.5% | 0.2% |
| Qwen3-8B, A/B log probabilities over both orders | 1,330 (66.50%) | 37.6% | 4.6% |
| Qwen3-8B, one seeded order | 1,320 (66.00%) | 36.9% | 4.9% |
| Jev 1.13.0, one choice question | 1,251 (62.55%) | 42.8% | 17.7% |
| Generic CLM head, default 2K | 1,212 (60.60%) | 94.7% | 73.5% |

The Qwen rotation adds 10 correct answers overall. The largest issue is the decision threshold: Qwen rarely calls an email phishing; CLM calls most emails phishing. These are uncalibrated default decisions. The URL rule does better than every model without training or an API call, indicating that this dataset is dominated by URL construction.

Jev's 1,251/2,000 is close to the upstream project's published 1,252/2,000, but the protocols differ. Their Jev call bundled nine questions; this run asks only the shared verdict question. The close scores do not establish that the two payloads are equivalent.

The category breakdown in [`summary.json`](../results/phishing/summary.json) makes the failure pattern concrete. For example, on 196 phishing emails with Google Docs links, Jev correctly identifies 3, rotated Qwen 4, and CLM 192. On 333 legitimate cross-domain emails, Jev correctly accepts 186, Qwen 333, and CLM 109. These examples explain why overall accuracy alone hides very different operating points.

Median measured time per email was 0.336 s for Jev's remote API, 0.067 s for the two Qwen local forwards, and 0.101 s for the three CLM local forwards. They include different network and GPU environments and are not a direct deployment speed comparison. Jev reported 1,094,888 input tokens for 2,000 requests. No CLM state exceeded the 2K cap.

The saved [Jev responses](../results/phishing/jev_2000.jsonl), [Qwen and CLM scores](../results/phishing/qwen_clm_2000_clm2k.jsonl), and [reverse-order Qwen scores](../results/phishing/qwen_rotated_2000.jsonl) contain IDs, labels, probabilities, timings, and input hashes, but no email text or API key. Recompute and validate the report with `python scripts/summarize_phishing.py` after preparing the upstream data.

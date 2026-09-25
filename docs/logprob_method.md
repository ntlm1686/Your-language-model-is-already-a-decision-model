# Next-token probability decision rule

![Qwen3.5-9B next-token log-probability decision diagram](assets/qwen_logprob.svg)

For each task, we write the state, question, and lettered candidate descriptions in a Qwen3.5-9B chat prompt. The prompt ends at `Answer: `. We do **not** ask the model to generate a response. One forward pass yields the next-token logits for the single-token letters `A`, `B`, and so on. We apply `log_softmax` across those candidate-letter logits, rather than across the full vocabulary.

We cyclically rotate the candidates so each semantic option occupies each letter position once. For every rotation, we run one model forward pass and add the log probability of the option's current letter to that option's running score. The prediction is the semantic option with the greatest **sum of log probabilities**. The three-option diagram shows one ordering with illustrative vocabulary probabilities: A 56%, B 24%, C 8%, and other tokens totaling 12%. Dashed rows represent tokens we ignore. Renormalizing only A/B/C gives 63.6%, 27.3%, and 9.1%, respectively. The implementation computes the equivalent distribution directly with log-softmax over the option-token logits; it need not calculate a full-vocabulary softmax. The diagram’s readable prompt omits the chat-template wrapper, and its values are not benchmark outputs. With *N* candidates, this procedure uses *N* forward passes for each decision.

This matches the implementation in [`scripts/qwen3_8b_letters.py`](../scripts/qwen3_8b_letters.py) and the tool-decision evaluation in [`scripts/run_tool_decisions.py`](../scripts/run_tool_decisions.py). In the latter, candidate descriptions may be shuffled when loading tasks; the score is still accumulated for the same semantic option across rotations.

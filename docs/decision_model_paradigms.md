# Possible decision model architectures

Decision model architectures mainly differ in how state and action are connected and interact within the network.

![Three decision-model scoring architectures](assets/decision_model_paradigms.svg)

Let **S** be the current state, **A₁…Aₙ** the candidate actions, **N** the number of candidates, and **T** the number of state tokens or visual features. The diagram shows where information from S enters each action score. Any pattern can accept images if its encoder, data, and training support them; the released implementations have different modality support.

| Pattern | Inference path | Training | Reuse and cost |
| --- | --- | --- | --- |
| **1. Next-token probabilities (Qwen3.5-9B)** | Put S and all A₁…Aₙ in one prompt; read the next-token A/B/… probabilities after `Answer:`. In our rotated-option evaluations, align each letter probability with its semantic action and sum across rotations. | No decision-specific training. | N full-prompt forwards for N rotations in the reference evaluator. The optimized server batches these rotations into one call. |
| **2. Contrastive similarity (CLM)** | Encode S and each Aᵢ independently; compare their projected vectors with scaled cosine similarity and choose the top action. | CLM uses bidirectional InfoNCE on state–action pairs. | Encode a new state once. Repeated action vectors can be cached; scoring takes N vector comparisons. |
| **3. Action-token cross-attention (RAM-inspired proposal)** | Encode A₁…Aₙ into the input token sequence of a Transformer. Separately encode S into T features. Inside each Transformer layer, action tokens supply the queries and state features supply keys and values to cross-attention. Residual connections, normalization, and feed-forward sublayers update the action tokens; the final tokens produce per-action logits. | Would require state–action supervision: softmax cross-entropy for one valid action or a multi-label loss for several. | Encode the state once. Cross-attention costs roughly N × T per layer; the updated action tokens depend on S and cannot be cached as final scores. This design has not been trained or evaluated in this repository. |

In pattern 3, action tokens are the Transformer’s input sequence and continue through its layers. The state encoder supplies a separate feature sequence to the cross-attention sublayers. The residual path belongs inside each Transformer layer; it adds the attention update to the action representation. The diagram abbreviates normalization placement and does not require action-token self-attention.

## Implemented models and reproduction

Pattern 1 is implemented in the [Qwen3.5-9B runner](../scripts/eval_qwen35_letters.py) and [Jev-format API](../scripts/serve_qwen35_jev.py); see the [letter-scoring diagram](logprob_method.md). Pattern 2 is the [released CLM](https://github.com/Contrastive-LM/CLM) used in the benchmarks. Pattern 3 is a design proposal. Released Open-Jev is a separately trained pairwise state–action scorer, described in the [Open-Jev reproduction note](open_jev.md#model-structure).

For a controlled future experiment, use the same states, candidate actions, split, and supervision for all three designs. Compare accuracy and latency as N and T vary to test whether state-conditioned action tokens justify their extra cost over cached CLM vectors.

## Primary sources

- [CLM repository](https://github.com/Contrastive-LM/CLM#how-it-works): independent state/action encoders, contrastive training, and similarity scoring.
- [Recognize Anything paper](https://openaccess.thecvf.com/content/CVPR2024W/MMFM/papers/Zhang_Recognize_Anything_A_Strong_Image_Tagging_Model_CVPRW_2024_paper.pdf) and [RAM repository](https://github.com/xinyu1205/recognize-anything): image tagging with label queries.
- [RAM implementation](https://github.com/xinyu1205/recognize-anything/blob/main/ram/models/ram.py): label embeddings enter a tagging decoder over image embeddings.

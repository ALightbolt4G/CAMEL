# MSLM: Context-Aware Multi-cell Emergent Language Model
*A Biologically-Inspired Sparse Architecture for High-Efficiency Neural Networks*

**Author:** Adham  
**Status:** Prototype — Components Validated Individually, End-to-End Integration Pending  

## Abstract
We present MSLM (Multi Small Language Models), a modular architecture that decomposes a monolithic Dense Transformer into specialized "Cells" (LoRA adapters on a shared backbone). MSLM employs a two-tiered Biological Router (Thalamus for keyword matching, Prefrontal Cortex for semantic similarity) and includes theoretical designs for a Concept Graph with Hebbian Learning and a Hippocampus memory module. Our experiments on `BLOOM-560m` with 4GB VRAM demonstrate that individual cells learn domain-specific distributions, and adapter switching produces distinct outputs without mutual interference. However, the Router's activation threshold mechanism failed to reliably activate cells in practice, and the full end-to-end pipeline has not yet been integrated. This paper reports both successes and failures transparently.

## 1. Introduction
Large Language Models (LLMs) activate all parameters for every query, regardless of the domain. MSLM proposes a modular alternative: a network of specialized LoRA adapters (Cells) on a frozen base model, activated selectively by a Biological Router. This paper documents the current state of the architecture — what works, what doesn't, and what remains theoretical.

## 2. Architecture: The Biological Router
The router is designed to decide which cells to activate.

### 2.1 Thalamus (Zero-Parameter Routing)
Using a Regex-based Keyword Matrix (Gene Map), the Thalamus scans the query for domain-specific patterns (e.g., "war", "python", "equation"). It produces an initial score per domain in $O(N)$ time with zero tensor operations. **Status: Implemented and tested.** Works well for unambiguous queries but fails on semantically complex, multi-domain inputs.

### 2.2 Prefrontal Cortex (Semantic Similarity)
Uses `all-MiniLM-L6-v2` (a 23M parameter SentenceTransformer) to compute cosine similarity between the query embedding and pre-defined domain signature embeddings. **Status: Implemented and tested.** Provides useful semantic signal, but its VRAM cost (~90MB) means the routing phase is not truly "zero VRAM" — only the Thalamus qualifies for that claim.

**Note:** A gating mechanism (bypass PFC if Thalamus confidence > 0.8) is described in the architecture but is not implemented in the current codebase.

### 2.3 Hippocampus (Context Memory)
Designed to maintain conversational memory using an Exponential Moving Average (EMA) with attention reset for topic switches. **Status: Implemented as standalone module.** Used only in the interactive demo (`demo.py`) for score adjustment. Never tested with actual model inference or measured for impact on routing accuracy.

## 3. The Concept Graph & Hebbian Learning
Cells are connected via an Adjacency Matrix $W$. Activation spreads via: $A_{t+1} = \max(0, \min(1, A_t + \alpha (W \cdot A_t)))$. Connections update using Hebbian Learning: $\Delta W = \eta \cdot C \cdot (1 - W) - \gamma \cdot W$ where $C = A_t A_t^T$.

**Status: Implemented and unit-tested.** However, Spreading Activation and Hebbian Learning are used only in `demo.py` (a terminal simulation). They have never been integrated into the real inference pipeline or tested with actual model outputs.

## 4. Aggregation & Weight Tying
Cell outputs are merged using a Weighted Sum. The Aggregator employs Weight Tying with the shared embedding layer to produce vocabulary logits. **Status: Implemented and unit-tested.** Never used in the real inference pipeline because the pipeline (`pipeline.py`) is not integrated — all forward methods are `pass` stubs.

## 5. Theoretical Performance Projections
To estimate the efficiency gains of sparse activation, we compared two randomly-initialized models (not the real MSLM system):
- **Dense Baseline:** ~15M parameter Transformer
- **Mock MSLM:** ~4.9M parameter single-cell Transformer

| Metric | Dense Model (~15M params) | Single Cell (~4.9M params) | Ratio |
|---|---|---|---|
| **Latency** | 18.07 ms / query | 6.01 ms / query | **3.00x** |
| **VRAM** | 67.02 MB | 22.38 MB | **66.6% smaller** |

> [!WARNING]
> These numbers are **theoretical projections** from `benchmark/evaluate.py` using toy models with random weights. They measure the architectural overhead difference between a 15M-parameter and 4.9M-parameter transformer, not the actual MSLM system performance. The real system uses BLOOM-560m (560M parameters) and was not benchmarked end-to-end.

## 6. Training Results
Fine-tuning on `bigscience/bloom-560m` using LoRA (`r=16, alpha=32, fp16`) on a 4GB VRAM Quadro M1200 GPU:

| Cell | Version | Start Loss | End Loss | Final Accuracy | Training Time | Data Size |
|---|---|---|---|---|---|---|
| **history_cell** | v1 | 3.327 | 3.212 | 0.3988 | 47 mins | 385 chunks |
| **math_cell** | v1 | 3.084 | 2.883 | 0.4471 | 21 mins | 197 chunks |
| **math_cell** | **v2** | **2.649** | **0.2995** | **0.9255** | **138 mins** | **7116 chunks** |
| **code_cell** | v1 | 2.722 | 2.591 | 0.4904 | 12 mins | 103 chunks |
| **rezero_cell** | v1 | 3.975 | 3.188 | 0.4280 | 104 mins | 4545 chunks |

These are real numbers from actual training runs.

### 6.1 Math Cell v2: Massive Data Scaling Experiment
The `math_cell` was retrained with 36x more data (197 → 7116 chunks) using a mixed dataset: synthetic algebra/calculus problems generated via SymPy, theorem Q&A, and descriptive text extracted from *Thomas' Calculus* and *The Art and Craft of Problem Solving* PDF textbooks.

**Training metrics improved dramatically:** Loss dropped from 2.649 to 0.2995 (perplexity 14.13 → 1.35), and token accuracy reached 92.55% (peak: 94.12%).

**However, these numbers require a critical caveat:** A training accuracy of 92.5% with a loss of 0.29 on a 560M model almost certainly indicates **overfitting**. The model memorized training data patterns rather than learning generalizable mathematical reasoning. This is confirmed by the output quality tests (Section 7), where the model fails to correctly answer basic questions it should have memorized. The likely explanation is that the model learned to predict tokens within the truncated 1200-character training chunks but cannot generalize to novel phrasings.

### 6.2 General Analysis
- **Code cell** achieved the lowest v1 loss despite the smallest dataset, likely because BLOOM's pre-training data already contains substantial code.
- **ReZero cell** started with the highest loss (3.975) due to fantasy content with Japanese names being far from BLOOM's distribution.
- **History cell** showed the smallest loss reduction (Δ = 0.115). The 385-chunk dataset was insufficient.
- Data volume alone does not guarantee output quality — the math_cell v2's high training accuracy did not translate to factually correct inference outputs.

## 7. Cell Output Quality (Honest Assessment)
Individual cells were tested with domain-specific questions. Results are mixed:

### 7.1 What Works
- **Domain-specific style transfer:** Each cell adopts a distinct output style. The Math cell uses "Definition 1.1" formatting. The Code cell produces function-like structures. The History cell references nations and dates. The ReZero cell generates fantasy dialogue with character names.
- **Adapter switching:** Loading multiple adapters and switching between them produces clearly different outputs for the same prompt, confirming that LoRA adapters function as intended behavioral switches.

### 7.2 What Fails
- **Factual accuracy is poor across all cells.** The History cell called Hitler "the leader of the Allied Army." The Math cell defined the Pythagorean theorem as "the law of continuity in number theory." The Code cell defined list comprehension as "a dictionary." These are severe hallucinations that make the outputs unreliable for any factual use.
- **Repetition loops:** The History cell frequently enters repetition ("The answer is not known. The question is what caused World War I. The answer is not known..."). This is a known failure mode of small models fine-tuned on tiny datasets without repetition penalties.
- **Shallow semantic understanding:** Cells learned surface-level patterns (formatting, vocabulary) but not deep understanding. The Code cell produces syntactic structures that look like code but are logically incorrect.

**Root cause:** BLOOM-560m is too small (560M params) to retain factual knowledge after LoRA fine-tuning with minimal data (100-4500 chunks). The cells learn style, not substance.

## 8. Router Performance (Honest Assessment)

### 8.1 Ranking Accuracy (What Partially Works)
Using `eval_router.py`, we tested whether the router assigns the **highest score** to the correct domain. The scoring formula is: $Final = (Thalamus \times 0.3) + (Prefrontal \times 0.7)$.

| Phase | Ranking Accuracy | Notes |
|---|---|---|
| Thalamus alone | 0/7 (0%) | Pure regex failed all semantic queries |
| Thalamus + PFC | 4/7 (57.1%) | Semantic embeddings helped significantly |
| + Action Verb Rule | 5/7 (71.4%) | Heuristic for "Write Python" → code_cell |

### 8.2 Activation Accuracy
We ran two rounds of network tests. The first round (v1, before Re:Zero integration) tested 7 complex multi-domain queries. The second round (v2, full network with all 4 cells) tested 8 queries of mixed complexity.

**Round 1 (v1): 0/7 activation success (0%)**
All 7 multi-domain queries failed to activate any cell. Every score was below the 0.2 threshold.

**Round 2 (v2): 5/8 activation success (62.5%)**

| Query | Active Cells | Top Score | Status |
|---|---|---|---|
| "What is the derivative of x squared?" | `[]` | math: 0.1418 | ❌ FAILED |
| "Who was Napoleon Bonaparte?" | `[]` | history: 0.1254 | ❌ FAILED |
| "Write a Python function to sort a list" | `[code_cell]` | code: 0.4098 | ✅ SUCCESS |
| "What happens to Subaru in Arc 3?" | `[rezero_cell]` | rezero: 0.4579 | ✅ SUCCESS |
| "Write a Python program to calculate derivative of x squared" | `[code_cell]` | code: 0.4766 | ✅ SUCCESS |
| "Calculate mathematical probability of WWI..." | `[]` | history: 0.1198 | ❌ FAILED |
| "Explain how recursion is similar to mathematical induction" | `[code_cell]` | code: 0.3147 | ✅ SUCCESS |
| "Write a Python simulation of WWII battle outcomes" | `[code_cell, history_cell]` | code: 0.3950, history: 0.3077 | ✅ SUCCESS (multi-domain!) |

**Key observations:**
- The router now successfully activates cells for queries with clear action verbs ("Write", "Explain") or strong domain keywords ("Subaru", "Python").
- **First successful multi-domain activation:** "Write a Python simulation of WWII battle outcomes" activated BOTH `code_cell` (0.3950) and `history_cell` (0.3077). This is the first empirical evidence of the router activating multiple cells for a single query.
- The router still fails on short, keyword-poor queries ("What is the derivative of x squared?", "Who was Napoleon Bonaparte?"). These produce scores in the 0.12-0.14 range — well below the 0.2 threshold.
- **The math_cell was never the primary activation** in any query, despite having the most data. The PFC embedding model associates mathematical terminology with programming more strongly than with pure math.

### 8.3 Previously Confirmed Routing (Re:Zero Integration)
- "In Arc 6, what happens to Subaru at the Pleiades Watchtower?" → `rezero_cell` (0.5283) ✅
- "Explain the history of the Witch of Envy" → `rezero_cell` (0.2560) ✅
- "Write a python script to simulate Return by Death loops" → `code_cell` (0.4524) ✅
- "The great war that destroyed the capital city was started by" → `history_cell` (0.3200) ✅

### 8.4 The Semantic Ambiguity Problem
Query: *"Calculate the statistical probability of WWI given European alliance mathematics"*

| Cell | Score |
|---|---|
| math_cell | 0.0586 |
| code_cell | 0.0839 |
| history_cell | 0.1141 |

All scores are far below the 0.2 threshold. The embedding model associates "Calculate" and "statistical" with programming more than pure math. The equal semantic distribution across domains prevents any single cell from activating. The 0.2 threshold correctly prevents false activation, but the system produces no useful output as a result.

## 9. Adapter Fusion Experiment
We loaded both `history_cell` and `rezero_cell` simultaneously and tested with an ambiguous prompt.

**Test Prompt:** *"Natsuki Subaru walked into the battlefield and saw"*

| Adapter | Output Summary | Observation |
|---|---|---|
| **History** | "...his platoon of Japanese troops marching to capture a field near Rangoun..." | Rejected fantasy context; pulled name into WWII Asian theatre |
| **ReZero** | "...that Emilia was about to make a huge mistake..." | Correctly identified character; generated fantasy dialogue |
| **Fusion (50/50)** | "...a group of soldiers at his side. The commander said..." | Generic military narrative; neither fully historical nor fantasy |

**Interpretation:**
- The two adapters produce genuinely different probability distributions for the same prompt. This is expected since LoRA adapters modify frozen backbone weights additively — by design, switching adapters changes the output distribution.
- The fusion result (linear weight averaging) is a well-known technique called "Model Soups" (Wortsman et al., 2022). The blended output is interpolation, not emergent behavior. It confirms that the adapter weights are compatible for linear combination, but this is a property of LoRA architecture generally, not a novel finding of MSLM.
- No catastrophic forgetting was observed, but this is trivially guaranteed by LoRA's frozen-backbone design. It would be notable only if the backbone weights were modified during training, which they were not.

## 10. Known Limitations

1. **Pipeline not integrated:** `pipeline.py` contains only `pass` stubs. The full end-to-end flow (Router → Graph → Cell → Aggregator) has never executed as a unified system.
2. **Router activation threshold too aggressive:** The 0.2 threshold prevents false positives but also prevents all activations on complex queries. Needs calibration or a different mechanism.
3. **Base model too small:** BLOOM-560m cannot retain factual accuracy after LoRA fine-tuning. Cells learn stylistic patterns, not knowledge.
4. **Training data too small:** 100-4500 chunks per domain is insufficient for robust specialization. Repetition and hallucination are direct consequences.
5. **Hippocampus, ConceptGraph, Hebbian Learning are untested in real inference:** These components exist as code but their impact on actual system performance is unknown.
6. **No standard benchmarks:** All evaluations are qualitative. No BLEU, ROUGE, perplexity, or domain-classification metrics were computed.

## 11. Future Work
1. **Integrate the full pipeline:** Connect Router → ConceptGraph → Cell Loading → Aggregator in a single forward pass.
2. **Calibrate router thresholds:** Use a validation set to find optimal activation thresholds per domain.
3. **Scale training data:** Move from hundreds to tens of thousands of chunks per cell.
4. **Add generation controls:** Repetition penalties, top-k sampling, temperature tuning.
5. **Upgrade base model:** Test with Phi-3 (3B) or Llama-3 (8B) on capable hardware.
6. **Implement standard benchmarks:** Measure perplexity, domain classification accuracy, and factual correctness systematically.
7. **Test Hippocampus + Hebbian Learning with real inference:** Measure their actual impact on multi-turn conversation quality.

## 12. Conclusion
MSLM demonstrates that modular LoRA-based specialization can produce domain-specific output styles from a shared backbone model. The Biological Router (Thalamus + Prefrontal Cortex) can rank domains with ~71% accuracy on unambiguous queries but fails to activate cells for complex multi-domain inputs. Individual components (Router, Cells, ConceptGraph, Aggregator) are implemented and unit-tested, but the full pipeline is not yet integrated. The project establishes a foundation for modular LLM architectures but requires significant work on router calibration, data scaling, and end-to-end integration before it can deliver on the promise of intelligent sparse activation.

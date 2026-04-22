# Cross-Lingual Jailbreak Defense: Extending JBShield to the Multilingual Setting

## Overview

Large language models (LLMs) are significantly more vulnerable to jailbreak attacks in non-English languages than in English. Prior work has established both the phenomenon and its mechanistic cause: safety alignment is primarily English-centric, and while the refusal direction is universal across languages, non-English inputs fail to generate sufficient signal along this shared direction — resulting in weaker harmful/harmless representation separation and higher jailbreak success rates.

However, no existing inference-time defense addresses this multilingual vulnerability. JBShield (USENIX Security '25) proposes an effective representation-level defense for English, but does not adapt to non-English inputs. This work extends JBShield to the multilingual setting by introducing a static language-adaptive scaling factor that compensates for the per-language signal deficit at inference time — without any model retraining.

---

## Gap Being Filled

| Work | Contribution | Limitation |
|---|---|---|
| MultiJail (ICLR '24) | Discovers cross-lingual jailbreak phenomenon | Solution requires fine-tuning |
| Refusal Direction is Universal (NeurIPS '25) | Identifies mechanism: weak harmful/harmless separation in non-English; refusal direction is universal | No defense proposed |
| JBShield (USENIX Security '25) | Inference-time defense via toxic/jailbreak concept manipulation | English only |
| **This work** | Extends JBShield to multilingual setting via static language-adaptive scaling | — |

---

## Core Mechanism (from prior work)

```
LLM refusal relies on a universal refusal direction (shared across languages)
                    ↓
Non-English inputs produce weaker toxic concept activation
→ Insufficient signal along the shared refusal direction
→ Weaker harmful/harmless separation (lower Silhouette Score)
                    ↓
Jailbreak succeeds more easily in non-English languages
```

**Key evidence from literature:**
- Refusal Direction is Universal: English Silhouette Scores consistently higher (e.g., 0.496 vs ~0.22 for non-English on Llama3.1-8B); refusal vectors are approximately parallel across languages
- MultiJail: unsafe rate rises from <1% (English) to up to 28% (low-resource languages) on ChatGPT

---

## Our Approach: JBShield-M (Multilingual)

At inference time, identify the input language and apply a pre-computed static scaling factor to amplify the toxic concept activation, compensating for the language-specific signal deficit along the universal refusal direction.

**Scaling factor computation (pre-computed from validation set):**
```
gap_lang = mean_projection(harmful, EN) - mean_projection(harmful, lang)
           [projection onto toxic concept vector vt]

α_lang   = 1 + λ · (gap_lang / gap_max)     # λ is a tunable hyperparameter
```

**Modified forward pass (extends JBShield-M eq. 12):**
```
Ĥ_lt = H_lt + (α_lang · δt) · vt
```

where `δt` is JBShield's original English scaling factor, `α_lang` amplifies it for non-English inputs proportionally to their activation gap.

---

## Experimental Design

### Preliminary Analysis (motivation validation, ~2 experiments)

Reproduce prior findings in our specific model/data configuration to confirm the intervention target is correct. Not claimed as novel contributions.

**P1 — Representation Separation Across Languages**
Replicate Silhouette Score analysis on our dataset/model setup to confirm harmful/harmless separation degrades in non-English settings.
- Input: `data/data/representation_dataset.json`
- Output: Silhouette Score per language, PCA scatter plots

**P2 — Toxic Concept Activation Gap**
Measure per-language toxic concept activation strength and projection onto the refusal direction. Compute the activation gap relative to English, which directly informs the scaling factor.
- Input: `data/data/representation_dataset.json`
- Output: Activation gap table per language; refusal direction projection values

---

### Main Experiments (primary contributions)

**Experiment 1 — Scaling Factor Computation and Validation**
Compute static per-language scaling coefficients from the validation set activation gaps. Validate that the scaling factor correlates with jailbreak success rate across languages (i.e., languages with larger gaps receive larger corrections).
- Input: Activation gaps from P2
- Output: Scaling coefficient table; correlation between gap magnitude and jailbreak success rate

**Experiment 2 — Jailbreak Defense Evaluation (Main Result)**
Evaluate JBShield-M against baselines across languages and models on jailbreak success rate (ASR) and false refusal rate on harmless prompts.

| Baseline | Description |
|---|---|
| No defence | Raw model without intervention |
| JBShield (English scaling only) | Original δt applied uniformly, no language adaptation |
| Refusal vector addition | Direct addition of the universal refusal direction (from Refusal Direction is Universal) |
| SELF-DEFENCE | Fine-tuning based multilingual defense (from MultiJail) |
| **JBShield-M (ours)** | Static language-adaptive toxic concept enhancement |

- Input: `data/data/jailbreak_dataset.json`, `data/data/representation_dataset.json`
- Output: ASR and false refusal rate per language per model

**Experiment 3 — Ablation: Effect of Language Adaptation**
Compare JBShield-M with and without the language-adaptive scaling (i.e., α_lang = 1 for all languages) to isolate the contribution of the multilingual extension.
- Output: Per-language ASR with and without adaptation

---

## Data

| Dataset | Languages | Size | Role |
|---|---|---|---|
| PolyRefuse | 13 + EN | 27k records | Representation analysis (harmful + harmless) |
| MultiJail | 10 | 442 prompts | Cross-lingual jailbreak evaluation |
| JBShield Data | EN | 65k records | English baseline + 9 attack types |

**Experiment data** (generated by `data/build_dataset.py`):

| File | Records | Used in |
|---|---|---|
| `data/data/representation_dataset.json` | 27,395 | P1, P2, Exp 1 |
| `data/data/jailbreak_dataset.json` | 68,350 | Exp 2, 3 |

**Focus languages:** English (baseline), Chinese, Arabic, Korean, Italian, Thai

---

## Models

| Model | Parameters | Notes |
|---|---|---|
| Llama-3-8B-Instruct | 8B | Used in all three reference papers |
| Qwen2.5-7B-Instruct | 7B | Strong multilingual coverage, especially Chinese |

---

## Expected Contributions

1. First inference-time, training-free defense against cross-lingual jailbreaks
2. Static language-adaptive scaling mechanism grounded in per-language activation gap analysis
3. Empirical validation that the multilingual vulnerability can be mitigated at the representation level without degrading normal response quality

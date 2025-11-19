# Ontology-Enhanced LLM Judges - Chunk 1: Complete Analysis

**Metadata:**
- Original file: ontology_enhanced_llm_judges.md
- Chunk: 1 of 3 (complete sections 1-4)
- Line range: 1-400 (approx)
- Context: TIR-Judge architecture analysis and ontology enhancement proposals

---

# Ontology-Enhanced LLM Judges: Architecture Analysis

*Based on Google's TIR-Judge paper (arXiv:2510.23038) and ontology-based enhancement proposals*

## Executive Summary

**TIR-Judge** (Tool-Integrated Reasoning Judge) from Google Research introduces reinforcement learning to teach LLM judges to write and execute Python code during evaluation. With only 8B parameters, TIR-Judge achieves 96% of Claude Opus-4's performance on complex listwise ranking tasks and outperforms text-only reasoning judges by 6.4% (pointwise) and 7.7% (pairwise).

**Key Innovation**: End-to-end RL framework that teaches judges **when** to invoke tools, **how** to interpret results, and **how** to chain reasoning with execution - not just providing tool access.

**Critical Finding**: Simply giving base models tool access shows **negative gains**. RL is essential to unlock tool-use capability.

**The Gap**: Despite breakthrough performance, TIR-Judge has four fundamental limitations that can be addressed through ontology and knowledge graph integration, creating a production-grade architecture suitable for regulated industries.

## The TIR-Judge Breakthrough (From Paper)

### Core Architecture

**Problem Setup**: Given user prompt `p` and model responses `{r₁, r₂, ..., rₙ}`, evaluate quality through:
- **Pointwise**: Assign scalar score to each response
- **Pairwise**: Select preferred response between two candidates
- **Listwise**: Rank N responses and identify best

**Tool-Augmented Trajectory**:
```
τ = [(t₁, c₁, e₁), (t₂, c₂, e₂), ..., (tₖ, cₖ, eₖ), pred]

where:
  tᵢ = natural language reasoning step
  cᵢ = generated Python code
  eᵢ = execution result from code interpreter
  pred = final judgment
```

**Iterative Process**:
```
τᵢ₊₁ = π_θ(p, {r₁,...,rₙ}, τᵢ)
eᵢ = Executor(cᵢ)
```

### Training Framework

**Three Core Principles**:

1. **Task Diversity**: Mix verifiable (math, code) and non-verifiable (chat, safety) domains
   - Teaches model **when** tool use is beneficial
   - Prevents over-reliance on code execution
   - 26K preference pairs across 6 domains

2. **Judgment Flexibility**: Support multiple evaluation formats
   - Pointwise scoring
   - Pairwise comparison  
   - Listwise ranking (3-5 candidates)

3. **Iterative RL without Distillation**: TIR-Judge-Zero bootstraps from base model
   - No teacher model required
   - Self-improvement through rejection sampling + RL cycles
   - Outperforms distilled variant by 1.2% at 4B scale

### RL Training Details

**Objective** (DAPO variant of GRPO):
```python
L(θ) = E[
    min(
        ρ(τ) * A(τ),
        clip(ρ(τ), 1-ε, 1+ε) * A(τ)
    )
] - β * KL(π_θ || π_ref)

where:
  ρ(τ) = importance weight
  A(τ) = advantage at token level
  ε = clipping range (0.2)
  β = KL penalty (0.01)
```

**Reward Structure**:
```python
R = R_correctness ∧ R_format ∧ R_tool

R_correctness = 1 if prediction matches ground truth, else 0

R_format = 1 if:
  - Scores in <score></score> tags
  - Preferences in <preference></preference> tags  
  - Code in ```python ... ``` blocks
  - (For chat/safety: no tool calls required)
  else 0

R_tool = 1 if:
  - ≤3 tool calls per trajectory
  - All code blocks execute without errors
  else 0

Final: R = 1 only when all three components = 1
```

### TIR-Judge-Zero: Iterative Self-Improvement

```
Initialize: θ₀ = base model checkpoint

Loop for k iterations:
  1. RL Training:
     θₖ = DAPO_RL(θₖ₋₁, D_train)
  
  2. Rejection Sampling:
     For each prompt p in D_train:
       Sample 8 trajectories from θₖ
       Keep trajectories where:
         - Produces correct answer
         - Satisfies output format
         - No interpreter errors
       Select shortest trajectory (or fewest tool calls)
     
     Build D_RS = {(p, τ_best)}
  
  3. Supervised Fine-Tuning:
     θₖ₊₁ = SFT(θₖ, D_RS)
     
  4. Validation:
     Evaluate θₖ₊₁ on held-out set
     Keep best checkpoint

Repeat until convergence (typically 2-3 iterations)
```

**Critical Implementation Details**:
- Mask interpreter outputs during loss computation (prevent overfitting)
- Truncate error messages to last line only (reduce context length)
- 8-gram decontamination against evaluation benchmarks
- Max response length: 8192 tokens
- Max tool calls per trajectory: 3

### Benchmark Results

**PPE Correctness (Reasoning)**:
| Model | Size | MMLU-P | MATH | GPQA | MBPP-P | IFEval | Avg |
|-------|------|--------|------|------|--------|--------|-----|
| Qwen3 | 8B | 67.0 | 72.4 | 54.0 | 56.0 | 34.0 | 56.7 |
| Qwen3+Tool | 8B | 67.0 | 72.4 | 54.0 | 56.0 | 34.0 | 56.7 |
| RRM-32B | 32B | 80.5 | 94.3 | 68.4 | 72.8 | 60.2 | 75.3 |
| **TIR-Judge-Zero** | 8B | **67.8** | **88.0** | **53.2** | **64.7** | **77.8** | **70.3** |

**Key Observations**:
1. Base model + tools = **no improvement** (even slightly negative)
2. RL unlocks tool-use capability (+13.6% average over base)
3. 8B TIR-Judge competitive with 32B text-only judges

**RewardBench2 (Listwise)**:
| Model | IF | Math | Fact | Focus | Safety | Avg |
|-------|----|----|------|-------|--------|-----|
| Claude Opus-4 | 41.9 | 74.9 | 82.7 | 86.2 | 89.5 | 76.5 |
| GPT-4.1 | 39.7 | 65.2 | 82.9 | 73.4 | 87.3 | 69.7 |
| **TIR-Judge-Zero 8B** | 45.6 | **84.1** | 64.8 | 89.5 | 82.7 | **73.4** |

**96% of Claude Opus-4 performance with 1/100th the parameters**

---

**End of Chunk 1**

**Next chunk preview (25 line overlap):**
Lines 376-400 will be repeated at the start of Chunk 2 to maintain context continuity.
# Does Stable Agentic RL Scale Down?

**NYU NLP course project · Team 4 (Yuxuan Feng, Yixuan Ma, Lilei Wu)**

We test ARLArena's finding on importance-sampling clipping at small scale: **which makes RL training more stable, tolerant token-level clipping or sequence-level clipping?**

> This repository started from [ARLArena](https://github.com/WillDreamer/ARL-Arena) (Wang et al., ICML 2026). All of our code is in [`course_project_grpo_clipping/`](course_project_grpo_clipping/), whose README documents the full experimental setup and how to run it. Our original paper analysis and project plan are in [`docs/project_planning.md`](course_project_grpo_clipping/docs/project_planning.md). The rest of the repository is ARLArena's original code, unmodified; see the [upstream repository](https://github.com/WillDreamer/ARL-Arena) for its own documentation. The `reference` branch is a snapshot of the upstream code.

---

## 1. What this project does

### Background

When large language models are trained with reinforcement learning (RL) for reasoning or agentic tasks, training often **collapses**: accuracy rises, then drops sharply, while KL divergence and gradient norms explode. ARLArena decomposes policy-gradient methods into four design dimensions and studies each one in isolation. It finds that **importance-sampling (IS) clipping is the dimension that matters most for stability** (Finding 1):

> tolerant clipping yields fast early gains but causes training collapse, whereas sequence-level clipping ensures stable improvement.

This result was obtained with 4B–8B models on H200/B200 GPUs, a scale most students and small labs cannot reproduce.

### Research question

Does ARLArena's Finding 1 still hold for a **0.5B model trained with LoRA on a single GPU**?

### Comparison and hypothesis

| | (a) Tolerant token-level clipping | (b) Sequence-level clipping (GSPO) |
|---|---|---|
| How it works | One ratio per token; wide upper clip bound (ε_high = 0.6) | One ratio per answer (geometric mean), clipped once |
| Prediction | Faster early gains, then collapse | Stable improvement |

**Hypothesis**: tolerant clipping will produce faster early improvement followed by collapse, while sequence-level clipping will be more stable.

### Method and evaluation

- **Method**: train Qwen2.5-0.5B-Instruct + LoRA on GSM8K with Hugging Face TRL's GRPOTrainer. The two variants differ **only in the clipping rule**; everything else is held fixed.
- **Evaluation**: held-out accuracy over training steps, plus stability signals: KL divergence, gradient norm, fraction of clipped tokens, and format validity. A run counts as "collapsed" if accuracy drops substantially from its peak together with a spike in KL or gradient norm.

---

## 2. What has been done

| Status | Item |
|---|---|
| ✅ | **Training pipeline**: TRL GRPOTrainer + LoRA, with switchable clipping granularity (token / sequence) and width (ε) |
| ✅ | **NYU Torch cluster setup**: Python environment with pinned dependencies, model and data pre-downloaded, Slurm launcher |
| ✅ | **Two smoke tests**: fixed environment issues; confirmed that `num_iterations = 2` is required, since otherwise the ratio is always 1 and clipping never fires |
| ✅ | **v1 pilot (4 runs × 150 steps)**: found **reward hacking** — the model stopped reasoning and only emitted answer tags. Fixed by switching to Qwen's native `\boxed{}` answer format |
| ✅ | **v2 pilot (4 runs × 300 steps, single seed)**: training behaves normally; results below |
| ✅ | Code, configs and per-step logs uploaded to this repository |
| ✅ | Midterm presentation |

### Main findings from the v2 pilot

| | (a) Tolerant (`token_wide`) | (b) Sequence-level (`seq_narrow`) |
|---|---|---|
| Held-out accuracy (step 50 → step 300) | 35.9% → 39.1% | 33.1% → 36.9% |
| Fraction of clipped tokens | 0.02% | 33.3% |
| Final KL | 0.0032 | 0.0022 |
| Collapsed? | No | No |

- **The pipeline is feasible**: each run takes about 1.1–1.4 hours on a single L40S.
- **Neither variant collapsed**, and the accuracy gap (about 3 points) is within noise. The hypothesis is **neither supported nor refuted** yet.
- **Clipping almost never fires in the tolerant variant**: in the current setting the policy changes very little per step (KL ≈ 0.003), so the wider bound is never reached.
- **To really test the hypothesis we need more off-policy updates**; this is the focus of the next step.

See [`course_project_grpo_clipping/README.md`](course_project_grpo_clipping/README.md) for detailed results and training curves.

---

## 3. What comes next

| Priority | Task | Purpose |
|---|---|---|
| 1 | **Stress test**: raise `num_iterations` to 4 / 8, or the learning rate to 5e-5 | Find a setting where clipping actually binds, and see whether the tolerant variant starts to collapse |
| 2 | **Multiple seeds**: 3 seeds per variant, report mean ± std | Separate real effects from noise (required by the proposal) |
| 3 | **Diagnose the collapse mechanism**: fraction of out-of-bounds IS ratios split by advantage sign; try sequence masking | Test the paper's claim that collapse is driven by accumulating negative-advantage, low-ratio sequences |
| 4 | **Add CISPO / SAPO-style tolerant variants**, based on the official implementations in `verl/trainer/ppo/core_algos.py` in this repository | ARLArena's "tolerant clipping" refers to CISPO/SAPO, which keep gradients of out-of-bounds tokens; our current variant only widens ε |
| 5 | **Model scale**: repeat key experiments with Qwen2.5-1.5B | Check whether the effect strengthens with scale (robustness section of the proposal) |
| 6 | *Stretch*: a tiny multi-turn Sokoban environment | Move from single-turn math to a genuinely multi-turn agentic setting |
| 7 | Final report | Summarize all results |

### Known limitations

- Only one seed per variant so far.
- About 15% of answers hit the 400-token limit and are truncated (same rate in both variants).
- 300 steps cover only about 8% of the training set.
- GSM8K is a single-turn task, not a multi-turn agentic setting.

---

## Citation

The original work this project reproduces and tests:

```bibtex
@article{wang2026arlarena,
  title={ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning},
  author={Wang, Xiaoxuan and Zhang, Han and Wang, Haixin and others},
  journal={arXiv preprint arXiv:2602.21534},
  year={2026}
}
```

ARLArena's code is distributed under its original [LICENSE](LICENSE).

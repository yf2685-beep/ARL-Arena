> **Note**: this is the project's original planning document (September 2026, written before the proposal), kept here for reference. Parts of it are out of date:
> - The experiments actually run on the **NYU Torch cluster (one L40S)**, not Colab.
> - The script sketch in Section 5 has been replaced by [`grpo_clipping_experiment.py`](../grpo_clipping_experiment.py).
> - For current progress and results, see the [repository README](../../README.md); for experimental details, see [`course_project_grpo_clipping/README.md`](../README.md).

---

# ARLArena: Paper Analysis and Course Project Plan

> Paper: *ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning*
> (Wang et al., ICML 2026, arXiv:2602.21534v3)
> Code: https://github.com/WillDreamer/ARL-Arena.git

This document analyzes the paper's research direction, its four-dimension decomposition framework, its contributions and its literature-review structure, and lays out a feasible NLP course project plan (direction: stability analysis / ablation reproduction; compute: Colab level).

---

## 1. Research direction (what it studies)

**Core topic: training stability in Agentic Reinforcement Learning (ARL).**

Background:

1. Large language models (LLMs) are increasingly used as **autonomous agents** for multi-step interactive tasks (web navigation, embodied environments, games, deep search, tool use).
2. Such tasks need **multi-turn** training objectives rather than single-turn QA. RL post-training is the mainstream approach (DeepSeek-R1 and OpenAI o1 are examples).
3. **But agentic RL (ARL) training is highly unstable and often collapses outright.** Multi-turn interaction compounds several difficulties: invalid actions, sparse rewards, long-horizon credit assignment, and non-stationary agent–environment dynamics. A small early deviation can cascade across turns, causing distribution drift and degenerate rollouts.

**The problem the paper addresses**: making ARL training stable, reproducible and scalable. Rather than proposing a "stronger model", it proposes an **analysis framework + a stable training recipe + a new algorithm**.

Concretely, three things:

- **ARLArena**: a standardized, reproducible multi-turn ARL testbed plus a systematic analysis methodology.
- **Decomposing the policy gradient into four orthogonal design dimensions**, and isolating the effect of each on stability and performance.
- **SAMPO** (Stable Agentic Multi-turn Policy Optimization): a unified policy-optimization method derived from the analysis, targeting the main sources of instability it identifies.

---

## 2. The four design dimensions (the paper's analytical backbone)

This is the paper's central organizing idea: all mainstream RL algorithms (GRPO, DAPO, GSPO, CISPO, SAPO, GiGPO, EMPG, etc.) are aligned and compared along **the same four dimensions** (see Table 1 of the paper):

| Dimension | What it studies | Key finding |
|---|---|---|
| **1. Loss aggregation** | token-mean vs. seq-mean-token-mean | Sequence-level aggregation hurts training on tasks with large length variation (e.g., math) because tokens are weighted unevenly |
| **2. Importance-sampling (IS) clipping** | tolerant clipping vs. sequence-level clipping | **The most sensitive dimension.** Tolerant clipping gains quickly early on but collapses; sequence-level clipping ensures stable improvement (Finding 1) |
| **3. Trajectory filtering / resampling** | filtering out zero-gradient trajectories | More stable when combined with GiGPO than with GRPO (Finding 4) |
| **4. Advantage design** | adding environment-level / global–local information | Fine-grained environment advantages alleviate reward sparsity and improve performance (Finding 3) |

Three core conclusions (stated in the abstract):

1. Tolerant clipping → collapse; sequence-level clipping → stability.
2. Adding environment-level information to the advantage design → better stability and performance.
3. Combining dynamic sampling with fine-grained advantage design → further gains.

The **root-cause diagnosis** of collapse is especially instructive: the authors find that collapse is driven mainly by the **accumulation of sequences with negative advantage and low importance ratio**, and they visualize this with token- and sequence-level IS analysis (the streamgraph in Figure 3). Their fix is **sequence masking** (masking out these harmful sequences).

---

## 3. Contributions (novelty)

The paper lists four contributions:

1. **A unified perspective**: a common taxonomy of policy-gradient and PO methods along four dimensions. This "survey-style reframing + experimental alignment" makes methods that were previously described in incompatible ways directly comparable.
2. **A standardized, reproducible testbed** and a methodology for diagnosing stability in multi-turn ARL (ARLArena).
3. **Task-robust findings and remedies**: diagnoses and fixes for common collapse patterns (e.g., sequence masking).
4. **A new algorithm, SAMPO**: it **unifies** sequence-level clipping, fine-grained advantages (global + local) and dynamic filtering in one framework, improves over the GRPO baseline by 25.2% on average, and trains stably on ALFWorld, WebShop, Sokoban and TIR Math.

An additional interesting experiment (Section 5.3): a small open model (Qwen3-4B) trained with SAMPO **outperforms multi-agent systems built on GPT-5.2 and o3** on agentic tasks. The argument is that on agentic tasks, principled RL training is worth more than elaborate inference-time engineering.

**Methodological features worth borrowing for a course project:**

- Instead of inventing new components, the paper **decouples, isolates and recombines**: it splits existing methods into orthogonal dimensions, ablates each dimension, then recombines the best choice from each into a new method. This is a very solid and "publishable" research pattern.
- It emphasizes **reproducibility and fair comparison** (a standardized testbed, plus a hyperparameter grid search for every method).

---

## 4. Literature-review structure

The paper has no standalone related-work section; related work is **woven into the Introduction and Table 1** (common in ML conference papers). The cited work falls into several groups:

- **Foundations of RL post-training**: PPO (Schulman 2017), InstructGPT/RLHF (Ouyang 2022), KL approximation (Schulman 2017 blog).
- **RL for reasoning**: DeepSeek-R1, OpenAI o1.
- **The PO algorithm family (the core comparison)**: GRPO, DAPO (Yu 2025), GSPO (Zheng 2025), CISPO (Chen 2025), SAPO (Gao 2025), GiGPO (Feng 2025), EMPG (Wang 2025).
- **Agentic RL frameworks / environments**: AgentGym, Agent-R1, AgentRL, RAGEN, SWEET-RL, verl (training framework).
- **Agent environments / benchmarks**: ALFWorld, WebShop, WebArena, Sokoban, TIR Math, ReAct, Voyager, API-Bank.
- **Recent stability work**: CLEANER (self-purifying trajectories), TIPS (turn-level reward shaping), T²PO, latent Poincaré shaping, and other 2026 work.

---

## 5. NLP course project plan

**Direction: stability analysis / ablation reproduction. Compute: Colab level (a few GPUs).**

### Scope: reproduce and verify one finding

Do not attempt the whole of ARLArena (4 environments + Qwen3-4B/8B + H200; Colab cannot run it). **Focus on one of the paper's four dimensions and verify one of its findings.** That is enough for a course project.

**First choice: reproduce Finding 1 (the IS-clipping dimension) — "tolerant clipping causes collapse vs. sequence-level clipping is stable"**

- This is the paper's **most sensitive dimension with the strongest signal**; a clear collapse-vs-stable contrast should be visible even at small scale, and the resulting curves are convincing.
- Collapse symptoms (reward suddenly dropping, KL / gradient norm exploding) also occur in small models and are easy to observe.

**Second choice: the loss-aggregation dimension — token-mean vs. seq-mean-token-mean**

- The simplest to implement (it only changes how the loss is normalized), but the signal is weaker and only shows up clearly on tasks with large length variation.

### A feasible tech stack (Colab level)

| Component | Recommendation |
|---|---|
| Model | Qwen2.5-0.5B or 1.5B-Instruct + **LoRA** (full-parameter training is infeasible) |
| Framework | **TRL**'s `GRPOTrainer` (lighter than verl, Colab-friendly), or verl in single-GPU mode |
| Task | One of: (1) **GSM8K / small math problems** (single-turn, easy-to-define reward; get the pipeline working first); (2) **a tiny Sokoban** (3×3 to 4×4 grid; multi-turn but with a small state space) |
| Variable | Change only the clipping rule: standard PPO clip (tolerant) vs. sequence-level clip (GSPO-style); keep everything else fixed |
| Metrics | success rate / reward, off-policy KL, gradient norm, fraction of out-of-bounds tokens — **plot them like the five panels of the paper's Figure 2** |

Get the full RL pipeline working on the math task first (the easiest route), then decide whether to add a multi-turn agent environment. Even a minimal Sokoban is noticeably harder to tune.

### Project narrative (how to write the report)

1. **Related work / literature review**: reuse the framework of the paper's Table 1, introducing GRPO/DAPO/GSPO/CISPO along the four dimensions (going deep on only one).
2. **Method**: formalize the dimension under study and present the two variants being compared.
3. **Experiments**: small model + small task; plot training dynamics and reproduce collapse vs. stability.
4. **Analysis (where the "novelty" comes from)**: a course project does not need a new algorithm. The contribution can be:
   - **verifying whether the paper's finding holds at a smaller scale** (a scale-down reproduction is valuable in itself);
   - or a small observation / variant (e.g., a different masking threshold, or a different task to test whether the conclusion transfers).

### An honest caveat

Even at small scale, multi-turn agentic RL **easily runs out of memory or takes a long time on Colab** (slow rollouts, costly environment interaction). So it is strongly recommended to **first establish the collapse-vs-stable contrast on a single-turn math task** as the safe core of the project, and treat the multi-turn agent environment as a stretch goal if time allows. That way the whole project does not stall because an environment cannot be set up.

---

## 6. Proposal requirements (assignment instructions)

Please submit your Project Proposal for this assignment. Your proposal should present an initial plan for your project (which you may revise as your work progresses) and address the aspects outlined below.

**Writing Template:**
Please use the ACL conference template for the project submissions: https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj

**Project Proposal Outline:**

1. **Why Important? (Motivation):**
   Explain why this problem matters. What real-world or academic gap does it address?

2. **Related Work:**
   Briefly summarize the current state of the field and key prior work. Identify gaps or limitations you intend to address.

3. **Research Question or Goals:**
   State your central research question or hypothesis. Or list your main objectives. What do you aim to achieve?

4. **Methodology:**
   Describe your planned methods, models, datasets, or frameworks.

5. **Evaluation:**
   Explain how you will evaluate your approach (metrics, experiments, qualitative analysis, etc.).

6. **Tangible Outcomes / Contributions:**
   Outline your expected contributions (e.g., new algorithm, dataset, analysis, system, model, theoretical insight).

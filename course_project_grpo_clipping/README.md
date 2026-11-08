# GRPO Clipping Study (NLP course project, Team 4)

Testing ARLArena's Finding 1 at small scale: **which makes GRPO training more stable, tolerant token-level clipping or sequence-level clipping?**

> This folder is our own implementation. It is built on Hugging Face TRL, runs on a single GPU, and does **not** depend on the rest of this repository (ARLArena's original code is built on verl and targets 8-GPU training with 4B models). ARLArena's loss implementations in `verl/trainer/ppo/core_algos.py` (GSPO around line 1015, SAPO around line 1448) serve as a reference.

## Research question and hypothesis

- **Comparison** (as in the proposal): (a) tolerant token-level clipping (wide upper clip bound) vs. (b) sequence-level clipping (GSPO)
- **Hypothesis**: tolerant clipping will produce faster early improvement followed by collapse, while sequence-level clipping will be more stable.
- **Controlled design**: the variants differ only in the clipping rule; model, data, reward, batch size, learning rate, KL coefficient and random seed are identical.

## Layout

```
course_project_grpo_clipping/
├── grpo_clipping_experiment.py   # training script (TRL GRPOTrainer + LoRA)
├── plot_pilot.py                 # plots training curves from the logs
├── pilot.sbatch                  # Slurm launcher for NYU Torch
├── docs/project_planning.md      # original paper analysis and project plan
└── results/
    ├── v2/   # main pilot: 4 runs × 300 steps (run_config.json, log_history.json per run) + v2_dynamics.png
    └── v1/   # first pilot: flawed reward design (see below), kept as a negative example
```

## Experimental setup

| Item | Setting |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` + LoRA (rank 16, alpha 32, dropout 0.05, on q/k/v/o) |
| Data | GSM8K (`openai/gsm8k`): 7,473 training problems (shuffled by seed); evaluation on the first 200 test problems |
| Prompt | "Please reason step by step, and put your final answer within \boxed{}." |
| Reward | 1 if the number in the last `\boxed{}` equals the reference answer; +0.1 if a parsable `\boxed{number}` is present |
| GRPO | 4 questions per step × 8 answers per question (temperature 1.0, up to 400 tokens); advantages normalized within each group |
| Off-policy | `num_iterations = 2`: each rollout batch is used for 2 updates (with TRL's default of 1, the ratio is always 1 and clipping has no effect) |
| Optimization | AdamW, lr 1e-5 (linear decay), KL coefficient β = 0.04, gradient clipping 1.0, 300 steps, bf16 |
| Evaluation | Every 50 steps on the 200 test problems; 8 sampled answers per problem, reporting mean accuracy (not greedy) |
| Environment | NYU Torch, one L40S; torch 2.14.1, TRL 0.23.1, transformers 4.57.6, peft 0.21.2 |

### Clipping variants

| Run | Meaning | Ratio granularity | ε_low / ε_high |
|---|---|---|---|
| **token_wide** | **(a) tolerant token-level** | token | 0.2 / 0.6 |
| **seq_narrow** | **(b) sequence-level (GSPO)** | sequence | 3e-4 / 4e-4 |
| token_narrow | extra: standard GRPO | token | 0.2 / 0.2 |
| seq_wide | extra: tolerant sequence-level | sequence | 3e-3 / 1e-2 |

Following the proposal, the main comparison is **token_wide vs. seq_narrow**; the other two runs are additional references.

## How to run (NYU Torch)

```bash
# 1. Environment (use uv's managed Python: the system Python lacks Python.h, so Triton fails to compile)
uv python install 3.12
uv venv --python-preference only-managed -p 3.12 venv && source venv/bin/activate
uv pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
uv pip install "trl==0.23.1" "transformers==4.57.6" "peft==0.21.2" "datasets>=3.0" accelerate matplotlib

# 2. Pre-download the model and data on the login node
export HF_HOME=$SCRATCH/.huggingface
python -c "from huggingface_hub import snapshot_download; from datasets import load_dataset; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct'); load_dataset('openai/gsm8k','main')"

# 3. Submit (first change --account in pilot.sbatch to your own)
mkdir -p logs
sbatch pilot.sbatch token_wide --output_root runs_v2
sbatch pilot.sbatch seq_narrow --output_root runs_v2
#    smoke test: sbatch --time=00:20:00 pilot.sbatch token_wide --max_steps 5 --eval_steps 5 --n_eval 16 --output_root smoke
#    list all variants: python grpo_clipping_experiment.py --list

# 4. Plot
python plot_pilot.py runs_v2/*
```

## Progress

| Round | What we ran | Outcome |
|---|---|---|
| Smoke test 1 | 5 steps | System Python lacked `Python.h`; switched to uv's managed Python |
| Smoke test 2 | 5 steps | Pipeline runs at about 16 s/step; clipping fires |
| v1 | 4 runs × 150 steps, `<think>/<answer>` format with partial format reward | **Reward hacking**: the model only emitted `<answer>number</answer>`; answer length fell from about 55 to 10 tokens and accuracy from about 10% to 1% |
| **v2** | 4 runs × 300 steps, `\boxed{}` answer format | Training behaves normally; results below |

### v2 results (single seed)

| Run | Train accuracy (first 30 steps → last 30 steps) | Held-out accuracy (step 50 → step 300) | Fraction clipped | Final KL |
|---|---|---|---|---|
| **token_wide** | 51.7% → 55.2% | 35.9% → 39.1% | 0.02% | 0.0032 |
| **seq_narrow** | 48.1% → 55.8% | 33.1% → 36.9% | 33.3% | 0.0022 |
| token_narrow | 50.6% → 58.8% | 36.6% → 39.7% | 0.06% | 0.0032 |
| seq_wide | 48.8% → 56.7% | 34.2% → 38.4% | 0.94% | 0.0029 |

![v2 training dynamics](results/v2/v2_dynamics.png)

**Current conclusions**
- The pipeline is feasible: each run takes about 1.1–1.4 hours on one L40S.
- **Neither variant collapsed**, and the held-out accuracy gap (about 3 points) is within noise (the standard error on 200 problems is about ±3.4 points). The hypothesis is neither supported nor refuted yet.
- Clipping almost never fires in the tolerant variant (0.02%): with only 2 updates per batch and lr 1e-5, the policy changes very little per step (KL ≈ 0.003), so the wider bound is never reached.
- About 33% of tokens are clipped in the sequence-level variant, which also has the lowest KL and the most conservative updates.
- Train-accuracy curves of all four runs move together because they share the same seed and therefore see the same problems at each step.

**Known limitations**
- Only one seed per run (seed 0); the proposal plans multiple seeds.
- About 15% of answers hit the 400-token limit and are truncated (same rate in both variants).
- 300 steps cover only about 8% of the training set.
- "Tolerant" here means a wider ε upper bound (as in the proposal); in ARLArena, tolerant clipping refers to CISPO/SAPO, which keep gradients of out-of-bounds tokens.

## Next steps

1. **Stress test**: increase off-policyness (`num_iterations` 4 / 8, or lr 5e-5) and check whether the tolerant variant starts to collapse.
2. In a setting where clipping actually binds, run 3 seeds per variant and report mean ± std.
3. Log the fraction of negative-advantage, low-ratio tokens to test the collapse mechanism proposed in the paper.
4. Add CISPO/SAPO-style tolerant variants, based on the official implementations in this repository.

## Notes

- `results/v1` was produced with the old reward; the current `grpo_clipping_experiment.py` is the v2 version. When plotting v1 with `plot_pilot.py`, the "Boxed-answer rate" panel label does not apply.
- Full hyperparameters for each run are in `results/*/<run>/run_config.json`; per-step metrics are in `log_history.json`.

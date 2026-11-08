# quad_check: is the pilot fine-tune local enough for κ?

Implements the "Pilot check" doc: train a LoRA SFT pilot of Qwen2.5-0.5B-Instruct on GSM8K, then
compare the measured KL(π₀ ‖ π_t) along the LoRA line W₀ + t·(α/r)BA with the quadratic prediction
½t²·dᵀFd on old-task and new-task prompts. r(t) = KL / (½t²dᵀFd); r ≈ 1 means the quadratic
(Fisher) regime holds.

## Pipeline

| Stage | Module | Output (under `out_dir`) |
|---|---|---|
| 1 prompts | `quad_check.prompts` | `data/*.jsonl`: SFT set, check sets, eval sets |
| 2 train | `quad_check.train_pilot` | `adapters/{step_25,step_50,step_75,final}`, `train_log.csv` |
| 3 eval | `quad_check.eval_acc` | `eval.json`: GSM8K / ARC / MMLU accuracy, base vs pilot |
| 4 rollouts | `quad_check.rollouts` | `rollouts_{new,old}.pt`: one π₀ sample per prompt (T=1, no truncation, no repetition penalty) |
| 5 check | `quad_check.check` | `check_<adapter>.json`: per-sequence KL(t), ½dᵀFd, aggregate r(t) with bootstrap CIs |
| 6 analyze | `quad_check.analyze` | `summary.md`, `kl_vs_t.png` |

Data: new task is GSM8K (train for SFT, 200 test prompts for the check). Old tasks are 70
ARC-Challenge, 70 MMLU (non-math subjects) and 60 IFEval prompts. All prompts use the Instruct
chat template, the same one used in training.

## Run locally (smoke test, ~2 min on a laptop GPU)

```bash
uv sync
for s in prompts train_pilot rollouts check eval_acc analyze; do
  uv run python -m quad_check.$s --config configs/smoke.yaml
done
uv run pytest -q tests          # CPU tests: padding, line scaling, r(t) -> 1 as t -> 0
```

## Run on Zaratan

```bash
rsync -a --exclude .venv --exclude outputs ./ zaratan:Algernon/experiments/quad_check/
ssh zaratan 'bash ~/Algernon/experiments/quad_check/slurm/setup_env.sh'   # login node, once
ssh zaratan 'sbatch ~/Algernon/experiments/quad_check/slurm/run_full.sbatch'
```

The venv, HF cache and outputs live in `/scratch/zt1/project/cmsc416/user/$USER/quad_check`.
The job runs every GPU stage in one allocation (1 GPU, 8 cores, 64 GB) and skips any stage whose
output exists, so resubmitting resumes where it stopped.

## Implementation notes (deviations from the doc's sketch)

- **position_ids from the attention mask.** For a plain forward, HF does not derive positions from a
  left-padded mask; without this, scoring sees different positions than sampling did.
- **Strict fp32** (TF32 off) for rollouts and the check.
- **Qwen's generation_config** sets T=0.7, top-p 0.8, top-k 20 and repetition penalty 1.05; all four
  are overridden so rollouts are true π₀ samples.
- **Stable KL**: KL = log E_p[exp(d − E_p d)] with d = z_t − z₀, via expm1/log1p. It agrees with the
  naive Σp(log p − log p_t) to about 6 significant figures, and stays accurate at small t.
- **Gate before reading results**: t=0 must equal the adapter switched off, and t=1 must equal the
  merged model, or the check aborts.
- **The code test is r(0.01) ≈ 1, not r(0.1) ≈ 1.** In the smoke run r(t) → 1 as t → 0 with
  r − 1 ∝ t, which is a real cubic term, and it already moves r(0.1) by 20–30%. The summary
  reports (r(0.1) − 1)/0.1 as a cubic coefficient.

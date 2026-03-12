# FDM Autoresearch Goal

You are optimizing hyperparameters for training a language model to read
Frequency-Division Multiplexed (FDM) signal tokens and answer reasoning questions.

## Task
A GPT-2 Medium model achieved these results with the config in `fdm_proxy_train.py`:
- Action accuracy: 98.3%
- Fact accuracy: 99.9%
- Meta accuracy: 99.7%

Your job is to find hyperparameter settings that achieve the **highest composite
accuracy** on the proxy eval (Stage 0 only, 2000 samples) for a NEW model
architecture: **Qwen3-0.6B-Base** or **LFM2.5-1.2B-Base**.

These models have different embedding spaces, tokenizers, and attention mechanisms
than GPT-2 Medium. The GPT-2 hyperparameters likely do NOT transfer optimally.

## What you may change in `fdm_proxy_train.py`
Search ONLY within these parameters (do not change model architecture or FDM encoding):

### Learning rate schedule
- `lr_stage0`: float in [1e-5, 1e-4]
- `warmup_ratio`: float in [0.0, 0.15]
- `lr_decay`: one of ["cosine", "linear", "constant"]

### Batch & gradient
- `batch_size`: one of [2, 4, 8]
- `grad_clip`: float in [0.5, 1.0, 2.0]
- `grad_accum_steps`: one of [1, 2, 4]

### Curriculum stage 0 config
- `stage0_epochs`: int in [3, 10]
- `stage0_samples`: one of [2000] (do not change — higher values timeout)
- `a_low`: float in [0.0, 0.05, 0.1]  (ASK contrast; lower = easier)

### Regularization
- `weight_decay`: float in [0.0, 0.01, 0.1]
- `dropout`: float in [0.0, 0.1]  (applied via model config if supported)

## Validation metric
Composite score (higher is better):
  score = 0.4 * action_accuracy + 0.4 * fact_accuracy + 0.2 * meta_accuracy

This is printed at the end of every training run as: `SCORE: X.XXX`

## Constraints
- Each experiment must complete in under 25 minutes on an A40 GPU (2000 samples = ~16 min; do not increase stage0_samples)
- Do NOT change: model name, FDM encoder logic, NUM_CHANNELS, tokenizer special tokens
- Do NOT change: evaluation logic or scoring formula
- One change per experiment — isolate variables

## Strategy hints
- Qwen3 uses a different tokenizer than GPT-2; embedding LR sensitivity may differ
- LFM2.5 has a larger hidden dim; may tolerate larger batch sizes
- Start by searching lr_stage0 (most impactful), then warmup_ratio, then batch_size
- If score plateaus, try a_low=0.0 (maximum contrast) to confirm signal readability

## New finding to investigate (Qwen3 only)
Qwen3 trained from scratch with 20K samples scored only 13-32% extra channel
accuracy across channel density configs, vs GPT-2's 28-95% at the same budget.
Yet Qwen3 fully trained (115K samples, 5-stage curriculum) achieves 100% extra accuracy.
The problem is data efficiency at low sample counts, not architecture.

Priority: stage0_samples is fixed at 2000 due to timeout constraints. Focus on lr_stage0, warmup_ratio, batch_size, and grad_clip instead.
- Search lr_stage0 first: try 1e-4, 5e-5, 1e-5
- Then warmup_ratio: try 0.0, 0.1
- Then batch_size: try 8, 2
- Always pair with a_low=0.0 (maximum contrast) first
- Only tune lr and warmup after establishing the learning threshold
- GPT-2 starts learning at ~5K samples; Qwen3 threshold is unknown but likely higher

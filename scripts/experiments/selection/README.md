# Selection Experiments

This directory contains the parameter-selection comparison experiment for MNIST
label-removal unlearning.

## Layout

- `selection_scheme_comparison.py`
  - Runs the full selector and ratio sweep for one trial.
- `results/`
  - Stores per-trial JSON outputs.

## Trial Convention

One `trial` means one full sweep over:

- selectors
- parameter ratios

with one fixed random seed.

To use two GPUs efficiently, run different trials on each GPU. Do not shard by
selector or ratio at this stage.

## CLI

The intended interface is just `trial` and `seed`.

GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/selection/selection_scheme_comparison.py \
  --trial 0 \
  --seed 0
```

GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/experiments/selection/selection_scheme_comparison.py \
  --trial 1 \
  --seed 1
```

Results are saved automatically under:

```text
scripts/experiments/selection/results/<model>/trial_<trial>_seed_<seed>.json
```

## Default Ratio Sweep

The default parameter-ratio grid is:

```text
0.01 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00
```

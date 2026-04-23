# Selection Experiments

This directory contains dataset-specific parameter-selection comparison experiments.

Current script:

- `selection_scheme_comparison_mnist.py`
  - MNIST label-removal unlearning
- `selection_scheme_comparison_pubmed.py`
  - PubMed-RCT20K text-transformer unlearning
- `selection_scheme_comparison_newsgroup.py`
  - Newsgroup text-transformer unlearning
- `selection_scheme_comparison_cifar10.py`
  - CIFAR-10 ResNet18 unlearning

Planned next scripts:

- `selection_scheme_comparison_cifar10.py`

## Layout

- `selection_scheme_comparison_mnist.py`
  - Runs the full selector and ratio sweep for one or more trials.
  - Dataset-specific runners should include the dataset name in the filename.
- `results/`
  - Stores per-trial JSON outputs and an aggregate JSON for each run.

## Evaluation Target

These runners do not select the best score over all update steps.

Instead, each selector/ratio pair is evaluated at the first step where:

- `self_acc <= target_self_acc`

The main comparison target is therefore:

- retain accuracy at the point where forgetting reaches the requested target

The default forgetting target is:

```text
target_self_acc = 0.1 (%)
```

## Trial Convention

One `trial` means one full sweep over:

- selectors
- parameter ratios

with one fixed random seed.

One command runs `num_trials` consecutive trials with seeds:

- `seed`
- `seed + 1`
- `...`
- `seed + num_trials - 1`

To use two GPUs efficiently, run different trials on each GPU. Do not shard by
selector or ratio at this stage.

## CLI

The intended interface is just `seed` and `num_trials`.

GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/selection/selection_scheme_comparison_mnist.py \
  --seed 0 \
  --num-trials 5
```

GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/experiments/selection/selection_scheme_comparison_mnist.py \
  --seed 5 \
  --num-trials 5
```

PubMed example:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/selection/selection_scheme_comparison_pubmed.py \
  --seed 0 \
  --num-trials 5
```

MNIST launcher for 5 jobs per GPU on GPUs `0,1`:

```bash
bash scripts/experiments/selection/launch_selection_mnist_5way.sh \
  --seed-start 0 \
  --num-trials 1
```

With extra runner args:

```bash
bash scripts/experiments/selection/launch_selection_mnist_5way.sh \
  --seed-start 0 \
  --num-trials 1 \
  -- --param-ratios 0.05 0.20 --selectors caps highest_k_outputs
```

Results are saved automatically under:

```text
scripts/experiments/selection/results/<model>/seed_<trial_seed>.json
```

and one aggregate file is also saved:

```text
scripts/experiments/selection/results/<model>/seed_<seed>_trials_<num_trials>.json
```

## Default Ratio Sweep

The default parameter-ratio grid is:

```text
0.01 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00
```

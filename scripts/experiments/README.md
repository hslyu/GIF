# Experiments

This directory is for experiment runners that are more paper- or ablation-oriented
than the reusable search/train entrypoints under `scripts/search` and `scripts/train`.

## Experiments

- `selection/`
  - Compare parameter selection schemes while keeping the influence update rule fixed.
  - Current target setup: MNIST label-removal style unlearning.
  - Update rule: generalized influence on the selected parameter subset.

## Reference material

The closest legacy experiment code lives in `GIF_reference/scripts`:

- `table1-parameter_selection_comparison_svhn_0.ipynb`
- `table1-parameter_selection_comparison_cifar10_0.ipynb`

Those notebooks compare selector families such as:

- `LowestKOutputs`
- `LowestKGradients`
- `HighestKOutputs`
- `HighestKGradients`
- `Random`

This repo also includes the newer `CAPS` selector, so the first rewritten experiment
extends the original comparison rather than reproducing it exactly.

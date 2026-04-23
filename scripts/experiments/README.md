# Experiments

This directory is for experiment runners that are more paper- or ablation-oriented
than the reusable search/train entrypoints under `scripts/search` and `scripts/train`.

## Experiments

- `influence/`
  - Compare influence/update schemes while keeping the MNIST edit protocol fixed.
  - Current runner:
    - `influence_scheme_comparison_mnist.py`
  - Reference notebook:
    - `GIF_reference/scripts/table2-3-IF_comparison_mnist.ipynb`
- `selection/`
  - Compare parameter selection schemes while keeping the influence update rule fixed.
  - Current runners:
    - `selection_scheme_comparison_mnist.py`
    - `selection_scheme_comparison_pubmed.py`
  - Planned runner: `selection_scheme_comparison_cifar10.py`
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

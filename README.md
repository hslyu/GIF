# GIF

Refactored codebase for the paper project on Generalized Influence Functions (GIF).

This repository is organized as a normal Python package so reviewers and readers can install it with `pip install -e .`, run tests, and inspect the reusable components without depending on old script-era path hacks.

## Status

The repository currently focuses on the reusable library code:

- influence methods and solver utilities
- model definitions
- dataset loader helpers
- parameter selection modules
- unit and integration tests

Current baseline note:
- `DataInf` is exposed as a LoRA-only baseline and is intended for `fcn_lora` experiments in the current repository state.

Experimental training and figure-generation scripts are intentionally excluded from this first refactor pass.

## Repository Layout

```text
GIF/
├── environment.yaml
├── pyproject.toml
├── README.md
├── docs/
├── src/gif/
│   ├── influence/
│   ├── data/
│   ├── models/
│   ├── selection/
│   ├── solvers/
│   ├── regularization.py
│   └── utils.py
├── scripts/
│   ├── search/
│   └── train/
└── tests/
    ├── influence/
    ├── integration/
    ├── selection/
    └── solvers/
```

## Installation

### Conda

```bash
conda env create -f environment.yaml
conda activate gif
```

### Development install

```bash
pip install -e .
```

## Testing

```bash
pytest
```

Slow integration checks stay opt-in through environment variables.

## Example API

```python
import torch
from gif.influence import compute_gradient, generalized_influence
from gif.models import LeNet

model = LeNet()
inputs = torch.randn(4, 3, 32, 32)
targets = torch.randint(0, 10, (4,))
loss = torch.nn.CrossEntropyLoss()(model(inputs), targets)

gradient = compute_gradient(model, loss)
```

## Notes for the paper release

- The code has been moved to a package-first layout under `src/gif/`.
- Legacy `python_path.sh`, notebook-driven workflows, and ad hoc entry scripts are not carried into this version.
- Once the experimental pipelines are stabilized, they can be added back under a dedicated `experiments/` or `scripts/` directory without polluting the library package.

## Acknowledgements

- Models adapted from `kuangliu/pytorch-cifar`
- Lanczos implementation informed by `noahgolmant/pytorch-hessian-eigenthings`

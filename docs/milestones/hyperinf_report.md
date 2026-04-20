# HyperINF Implementation Report

## Summary
`HyperINF` was implemented as a **repo-native inverse-approximation baseline** for this repository.

This implementation does **not** attempt to reproduce the upstream LoRA/GFIM pipeline.
Instead, it reuses the repository's existing restricted-operator setup and adds a
Schulz-style inverse solver that can be compared against GIF and TracIn in the same
unlearning framework.

---

## What was implemented

### Source
- `src/gif/solvers/hyperinf.py`
  - `hyperinf_inverse(...)`
- `src/gif/influence/hypeinf.py`
  - `hyperinf_update(...)`
  - `HyperInfluence`
  - `HypeInf` alias

### Script integration
- `scripts/search/_mnist_unlearning_common.py`
  - added `hyperinf` method branch
- `scripts/search/search_mnist_model.py`
  - added `hyperinf` scheme
  - added `--hyperinf-beta-scale`
- `scripts/search/run_gif_unlearning_mnist.py`
  - added `hyperinf` scheme
  - added `--hyperinf-beta-scale`
- `scripts/search/search_gif_unlearning_mnist.py`
  - added `hyperinf` scheme
  - added `--hyperinf-beta-scale`

### Tests
- `tests/solvers/test_hyperinf_solver.py`
- `tests/influence/test_hyperinf_influence.py`
- updated `tests/integration/test_tracin_script_integration.py`
  - now checks `hyperinf` CLI exposure too

---

## Design notes

### Solver shape
The solver uses a Schulz-style doubling recurrence for inverse approximation on a
matrix-free operator:

- initialize with a scaled identity-like step `beta`
- maintain a residual operator `R(v) = v - beta A(v)`
- update the solution with `x <- x + R_k(x)`
- square the residual operator each iteration

This makes the implementation compatible with:
- restricted subset operators
- matrix-free `H_J^T H_J`
- current `influence` / `solvers` repository split

### Method shape
The method wrapper reuses the same restricted system used by GIF:

- `g_full = compute_gradient(model, target_loss)`
- `rhs = H_J^T g`
- `A(x) = H_J^T H_J x`

Then it solves `A x = rhs` with the HyperINF-style solver.

---

## Special findings

### 1. HyperINF iteration count must stay small
This implementation uses a residual-operator squaring scheme.
That means the effective operator depth grows as:

- iteration 1 -> depth 1
- iteration 2 -> depth 2
- iteration 3 -> depth 4
- iteration 4 -> depth 8
- iteration 5 -> depth 16

So unlike LiSSA-style methods, large `max_iter` values are not appropriate here.
In practice, the useful range in this repository was:

- `max_iter=4`
- sometimes `max_iter=5` or `6`

This is the main behavioral difference that must be remembered when comparing
HyperINF to GIF.

### 2. This is a HyperINF-style baseline, not a faithful LoRA reproduction
The upstream HyperINF code path is tightly tied to:

- DataInf-based code reuse
- LoRA assumptions
- LLM/VLM workflows

This repository implementation intentionally does **not** reproduce those pieces.
It only ports the inverse-approximation idea into the current benchmark.

### 3. The method is usable for model edit
Even with the simplified repo-native formulation, the method produced non-trivial
unlearning updates on real MNIST checkpoints.

---

## Validation results

### Unit / integration tests
- HyperINF-specific tests:
  - `10 passed`
- Full test suite:
  - `35 passed, 3 skipped`

### FCN smoke run
Command:

```bash
python3 scripts/search/search_mnist_model.py \
  --model fcn \
  --checkpoint checkpoints/mnist_fcn_deep.pth \
  --schemes hyperinf \
  --device cpu \
  --batch-size 128 \
  --num-workers 0 \
  --num-target-samples 32 \
  --num-retain-batches 1 \
  --param-ratios 0.01 \
  --tol-grid 1e-4 \
  --max-iters-grid 4 \
  --edit-scale 0.02 \
  --save-json tmp/hyperinf_fcn_smoke.json
```

Observed best:
- retain_acc = `98.08%`
- self_acc = `99.39%`
- score = `0.0122`

This confirmed the script path and output schema, but forgetting strength was weak.

### ResNet18 HyperINF unlearning smoke
Command:

```bash
python3 scripts/search/run_gif_unlearning_mnist.py \
  --checkpoint checkpoints/tracin_resnet18_20ep.pth \
  --schemes hyperinf \
  --device cuda \
  --batch-size 128 \
  --num-workers 0 \
  --num-target-samples 64 \
  --num-retain-batches 1 \
  --param-ratio 0.003 \
  --tol 1e-4 \
  --max-iter 4 \
  --hyperinf-beta-scale 0.9 \
  --edit-scale 0.05 \
  --max-update-steps 12
```

Observed best:
- before retain_acc = `99.21%`
- before self_acc = `99.69%`
- best retain_acc = `98.61%`
- best self_acc = `86.94%`
- best score = `0.2307`

This confirms the method can perform real model-edit unlearning in the current
pipeline.

### Same-framework benchmark smoke
Command:

```bash
python3 scripts/search/search_mnist_model.py \
  --model resnet18 \
  --checkpoint checkpoints/tracin_resnet18_20ep.pth \
  --trajectory-dir checkpoints/tracin_resnet18_20ep \
  --schemes caps tracin hyperinf \
  --device cuda \
  --batch-size 128 \
  --num-workers 0 \
  --num-target-samples 64 \
  --num-retain-batches 1 \
  --param-ratios 0.003 \
  --tol-grid 1e-4 \
  --max-iters-grid 4 \
  --edit-scale 0.05 \
  --save-json tmp/hyperinf_benchmark_resnet18.json
```

Observed best:
- `caps`
  - retain_acc = `98.00%`
  - self_acc = `98.47%`
  - score = `0.0301`
- `tracin`
  - retain_acc = `98.40%`
  - self_acc = `47.24%`
  - score = `0.6869`
- `hyperinf`
  - retain_acc = `98.45%`
  - self_acc = `62.24%`
  - score = `0.5458`

This confirms that GIF-family baseline methods and HyperINF can now be compared in
the same benchmark schema.

---

## Conclusion

The HyperINF implementation is complete enough for this repository's benchmark use:

- solver implemented
- influence wrapper implemented
- script integration completed
- phase-level tests added
- final unlearning smoke completed

The main caveat is that this implementation should be treated as a
**HyperINF-style restricted inverse baseline**, not as a faithful reproduction of the
original LoRA-oriented HyperINF system.

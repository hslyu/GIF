# DataInf Report

## Summary
A **LoRA-only DataInf baseline** was implemented for FCN models.

This repository now exposes `DataInf` only in its parameter-efficient regime:

- base FCN weights are frozen
- LoRA adapter parameters are trainable
- DataInf-style update is computed only in adapter parameter space

---

## What was implemented

### Source
- `src/gif/models/lora.py`
  - `LoRALinear`
  - `LoRAFullyConnectedNet`
  - trainable-parameter vector helpers
- `src/gif/influence/datainf.py`
  - `datainf_update(...)`
  - `DataInfluence`

### Script integration
- `scripts/train/train_deep_fcn_lora_mnist.py`
- `scripts/search/search_deep_fcn_lora_mnist.py`
- generic support added to:
  - `scripts/train/train_mnist_model.py`
  - `scripts/search/search_mnist_model.py`
  - `scripts/search/_mnist_unlearning_common.py`

### Tests
- `tests/models/test_lora_fcn.py`
- `tests/influence/test_datainf_lora_influence.py`
- updated `tests/integration/test_tracin_script_integration.py`

---

## Design notes

### Why FCN-only
This pilot is intentionally limited to FCN because:
- FCN is all-Linear
- `Linear LoRA` is simple to insert
- it validates the adapter-space idea with minimal repository disruption

ResNet/Conv-LoRA is a separate engineering step and should be treated separately.

### Why this is closer to DataInf
This implementation is closer to DataInf's natural use case because:
- the editable parameter space is small
- the update lives only in adapter parameters
- the approximation remains cheap

---

## Validation results

### Tests
- LoRA / DataInf targeted tests:
  - `8 passed`

### FCN LoRA smoke training
Command:

```bash
python3 scripts/train/train_deep_fcn_lora_mnist.py \
  --device cpu \
  --epochs 1 \
  --batch-size 128 \
  --num-workers 0 \
  --max-train-batches 2 \
  --max-val-batches 1 \
  --save-path tmp/fcn_lora_smoke.pth \
  --trajectory-dir tmp/fcn_lora_smoke_traj \
  --base-checkpoint checkpoints/mnist_fcn_deep.pth
```

Observed:
- train_loss `0.0008`
- train_acc `100.00%`
- val_acc `98.44%`
- test_acc `98.12%`

This confirms the FCN LoRA training path works and preserves the strong base model.

### DataInf unlearning smoke
Command:

```bash
python3 scripts/search/search_deep_fcn_lora_mnist.py \
  --checkpoint tmp/fcn_lora_smoke.pth \
  --schemes datainf \
  --device cpu \
  --batch-size 128 \
  --num-workers 0 \
  --num-target-samples 32 \
  --num-retain-batches 1 \
  --param-ratios 0.01 \
  --tol-grid 1e-4 \
  --max-iters-grid 4 \
  --edit-scale 5.0 \
  --datainf-damping 1e-6 \
  --save-json tmp/datainf_fcn_lora_final.json
```

Observed best:
- before retain_acc = `98.10%`
- before self_acc = `99.29%`
- best retain_acc = `95.67%`
- best self_acc = `63.67%`
- best score = `0.5266`

---

## Interpretation

The LoRA-only `DataInf` baseline is **feasible**:
- the LoRA model path works
- the adapter-space DataInf update works
- the unlearning benchmark path runs end-to-end

But the tradeoff is still limited:
- forgetting is achievable with a strong edit scale
- retain accuracy drops noticeably as the edit becomes aggressive

So the correct conclusion is:
- `DataInf` should remain LoRA-only in this repository
- FCN LoRA support is sufficient for a benchmark baseline
- broader LoRA support would require separate Conv-LoRA / ResNet-LoRA work

The next realistic extension would be:
- stronger FCN LoRA tuning
- then Conv-LoRA / ResNet LoRA support

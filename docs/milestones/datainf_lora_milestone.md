# DataInf Milestone Plan

## Goal
Implement `DataInf` as a **LoRA-only baseline** that is closer to DataInf's natural
parameter-efficient setting.

This pilot is intentionally scoped to:
- `FCN` only
- `Linear LoRA` only
- `DataInf` update in LoRA parameter space only

The purpose is to test whether a LoRA-space formulation is feasible inside this
repository before attempting a wider ResNet / Conv-LoRA rollout.

---

## Scope

Included:
- FCN LoRA module
- LoRA-aware FCN training path
- LoRA-space DataInf update
- FCN LoRA unlearning/search smoke

Excluded:
- ResNet LoRA
- Conv2d LoRA
- trajectory-based LoRA methods
- fully generic adapter framework

---

## Progress Checklist

- [x] M1. FCN LoRA Module
- [x] M2. DataInf Update
- [x] M3. Train/Search Integration
- [x] M4. FCN LoRA Smoke Training
- [x] M5. FCN LoRA Unlearning Validation

---

## M1. FCN LoRA Module

Implemented:
- `LoRALinear`
- `LoRAFullyConnectedNet`
- trainable-parameter vector helpers
- base-state loading helper

Tests:
- base output equivalence at zero adapter
- trainable parameter count reduction

---

## M2. DataInf Update

Implemented:
- `datainf_update(...)`
- `DataInfluence`

Behavior:
- works only on trainable LoRA parameters
- uses a cheap diagonal proxy in adapter parameter space

Tests:
- update shape test
- target-loss direction test
- details payload test

---

## M3. Train/Search Integration

Implemented:
- `scripts/train/train_deep_fcn_lora_mnist.py`
- `scripts/search/search_deep_fcn_lora_mnist.py`
- generic `train_mnist_model.py` / `search_mnist_model.py` support for `fcn_lora`
- `datainf` method branch in `_mnist_unlearning_common.py`

Tests:
- script `--help` smoke
- generic search help includes `fcn_lora` and `datainf`

---

## M4. FCN LoRA Smoke Training

Executed:
- base FCN checkpoint loaded
- LoRA adapter trained for a short smoke run
- checkpoint writing confirmed

Observed:
- training path works
- FCN LoRA checkpoint loads successfully in search path

---

## M5. FCN LoRA Unlearning Validation

Executed:
- `datainf` search smoke on trained FCN LoRA checkpoint

Observed:
- the path runs end-to-end
- forgetting signal exists but is weak in the current pilot setup

Interpretation:
- feasibility is confirmed
- `DataInf` is now LoRA-only in this repository
- strength is still weak enough that broader LoRA support should be treated as future work

# Influence Scheme Comparison

This experiment directory benchmarks influence-style model editing methods on MNIST.

Current runner:

- `influence_scheme_comparison_mnist.py`
- `influence_scheme_comparison_cifar10.py`
- `influence_scheme_comparison_pubmed.py`
- `influence_scheme_comparison_newsgroup.py`
- `influence_scheme_comparison_svhn.py`

Target methods:

- `classical_if`
- `second_order_if`
- `tracin`
- `hypeinf`
- `datainf`
- `freezing`
- `ekfac`
- `gif`

Implementation notes:

- The experiment is rewritten as a standalone Python script.
- It uses the current `gif.influence` APIs instead of reusing notebook code.
- `gif` and `freezing` use `highest_k_gradients` for subset selection.
- `datainf` applies a diagonal inverse-curvature style update over all trainable parameters.
- `tracin` requires a trajectory directory with `epoch_*.pth` checkpoints.
- Results are saved under `results/<model>/`.
- Text benchmarks use the same method set on `TextTransformerClassifier` checkpoints trained by
  `train_pubmed_rct20k_transformer.py` and `train_newsgroup_transformer.py`.

import torch

import gif
from gif.models import LeNet
from gif.utils import prepare_model


def test_package_exports_core_api():
    assert callable(gif.compute_gradient)
    assert callable(gif.freeze_influence)
    assert callable(gif.second_influence)


def test_prepare_model_uses_packaged_model_registry():
    model = prepare_model("LeNet")

    assert isinstance(model, LeNet)


def test_lenet_forward_shape():
    model = LeNet()
    inputs = torch.randn(2, 3, 32, 32)

    outputs = model(inputs)

    assert outputs.shape == (2, 10)

from pathlib import Path

import numpy as np
import torch

from gif.influence import TracIn, compute_gradient, tracin_update_from_checkpoints
from gif.models import FullyConnectedNet
from gif.selection import HighestKGradients


def _save_linear_checkpoint(
    path: Path,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epoch: int,
    lr: float,
) -> None:
    model = torch.nn.Linear(2, 1, bias=True).double()
    with torch.no_grad():
        model.weight.copy_(weight)
        model.bias.copy_(bias)

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "net": model.state_dict(),
            "epoch": epoch,
            "lr": lr,
        },
        path,
    )


def _load_paths(tmp_path: Path) -> list[Path]:
    trajectory_dir = tmp_path / "traj"
    _save_linear_checkpoint(
        trajectory_dir / "epoch_001.pth",
        torch.tensor([[0.2, -0.1]], dtype=torch.float64),
        torch.tensor([0.05], dtype=torch.float64),
        epoch=1,
        lr=0.2,
    )
    _save_linear_checkpoint(
        trajectory_dir / "epoch_002.pth",
        torch.tensor([[0.4, -0.3]], dtype=torch.float64),
        torch.tensor([0.02], dtype=torch.float64),
        epoch=2,
        lr=0.1,
    )
    return [trajectory_dir / "epoch_001.pth", trajectory_dir / "epoch_002.pth"]


def test_tracin_update_has_expected_full_parameter_shape(tmp_path: Path):
    checkpoint_paths = _load_paths(tmp_path)
    model = torch.nn.Linear(2, 1, bias=True).double()
    criterion = torch.nn.MSELoss()

    target_inputs = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    target_targets = torch.tensor([[0.3]], dtype=torch.float64)

    update = tracin_update_from_checkpoints(
        model=model,
        checkpoint_paths=checkpoint_paths,
        target_inputs=target_inputs,
        target_targets=target_targets,
        criterion=criterion,
        device="cpu",
    )

    assert update.shape == (3,)


def test_tracin_update_moves_target_loss_upward(tmp_path: Path):
    checkpoint_paths = _load_paths(tmp_path)
    model = torch.nn.Linear(2, 1, bias=True).double()
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.35, -0.25]], dtype=torch.float64))
        model.bias.copy_(torch.tensor([0.01], dtype=torch.float64))

    target_inputs = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    target_targets = torch.tensor([[0.3]], dtype=torch.float64)

    update = tracin_update_from_checkpoints(
        model=model,
        checkpoint_paths=checkpoint_paths,
        target_inputs=target_inputs,
        target_targets=target_targets,
        criterion=criterion,
        device="cpu",
    )

    before_loss = criterion(model(target_inputs), target_targets).item()
    gradient = compute_gradient(model, criterion(model(target_inputs), target_targets))
    assert torch.dot(gradient, update).item() > 0

    with torch.no_grad():
        flat = torch.nn.utils.parameters_to_vector(model.parameters())
        flat = flat + 1e-4 * update
        torch.nn.utils.vector_to_parameters(flat, model.parameters())

    after_loss = criterion(model(target_inputs), target_targets).item()
    assert after_loss > before_loss


def test_tracin_update_can_be_projected_to_selector_subset():
    torch.manual_seed(0)
    model = FullyConnectedNet(
        input_size=4,
        hidden_size=8,
        output_size=3,
        num_layers=3,
        dropout_prob=0.0,
    ).double()
    criterion = torch.nn.CrossEntropyLoss()

    selector = HighestKGradients(model, ratio=1.0)
    selector.register_hooks()

    probe_inputs = torch.randn(4, 4, dtype=torch.float64)
    probe_targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    probe_loss = criterion(model(probe_inputs), probe_targets)
    probe_loss.backward()

    try:
        index_list = selector.get_parameters()
        assert len(index_list) > 0

        parameter_dtype = torch.nn.utils.parameters_to_vector(model.parameters()).dtype
        full_update = torch.arange(
            1,
            sum(parameter.numel() for parameter in model.parameters()) + 1,
            dtype=parameter_dtype,
        )
        subset_update = full_update[torch.as_tensor(index_list, dtype=torch.long)]

        before = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
        selector.update_network(subset_update)
        after = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    finally:
        selector.remove_hooks()

    assert not torch.allclose(before, after)


def test_tracin_update_class_api_runs_on_saved_trajectory(tmp_path: Path):
    checkpoint_paths = _load_paths(tmp_path)
    trajectory_dir = checkpoint_paths[0].parent
    model = torch.nn.Linear(2, 1, bias=True).double()
    criterion = torch.nn.MSELoss()
    tracin = TracIn()

    update = tracin.compute_update(
        model=model,
        trajectory_dir=trajectory_dir,
        target_inputs=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        target_targets=torch.tensor([[0.1]], dtype=torch.float64),
        criterion=criterion,
        device="cpu",
        index_list=np.array([0, 2], dtype=int),
    )

    assert update.shape == (2,)
    assert torch.isfinite(update).all()


def test_tracin_update_restores_model_state_after_checkpoint_scan(tmp_path: Path):
    checkpoint_paths = _load_paths(tmp_path)
    model = torch.nn.Linear(2, 1, bias=True).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.9, 0.8]], dtype=torch.float64))
        model.bias.copy_(torch.tensor([0.7], dtype=torch.float64))
    original = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    _ = tracin_update_from_checkpoints(
        model=model,
        checkpoint_paths=checkpoint_paths,
        target_inputs=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        target_targets=torch.tensor([[0.3]], dtype=torch.float64),
        criterion=torch.nn.MSELoss(),
        device="cpu",
    )

    restored = model.state_dict()
    for name, tensor in original.items():
        assert torch.equal(restored[name], tensor)

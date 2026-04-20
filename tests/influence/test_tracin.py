from pathlib import Path

import torch

from gif.influence import (
    TracIn,
    compute_gradient,
    load_tracin_checkpoint_paths,
    tracin_score_from_checkpoints,
)


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


def _manual_tracin_score(
    checkpoint_paths: list[Path],
    source_inputs: torch.Tensor,
    source_targets: torch.Tensor,
    target_inputs: torch.Tensor,
    target_targets: torch.Tensor,
) -> float:
    model = torch.nn.Linear(2, 1, bias=True).double()
    criterion = torch.nn.MSELoss()
    total_score = 0.0

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["net"])

        source_loss = criterion(model(source_inputs), source_targets)
        source_grad = compute_gradient(model, source_loss)

        target_loss = criterion(model(target_inputs), target_targets)
        target_grad = compute_gradient(model, target_loss)

        total_score += float(checkpoint["lr"]) * torch.dot(source_grad, target_grad).item()

    return total_score


def test_load_tracin_checkpoint_paths_sorts_epochs_and_ignores_best(tmp_path: Path):
    trajectory_dir = tmp_path / "traj"
    trajectory_dir.mkdir()

    _save_linear_checkpoint(
        trajectory_dir / "epoch_010.pth",
        torch.tensor([[0.3, -0.1]], dtype=torch.float64),
        torch.tensor([0.05], dtype=torch.float64),
        epoch=10,
        lr=0.1,
    )
    _save_linear_checkpoint(
        trajectory_dir / "epoch_002.pth",
        torch.tensor([[0.1, -0.2]], dtype=torch.float64),
        torch.tensor([0.03], dtype=torch.float64),
        epoch=2,
        lr=0.2,
    )
    _save_linear_checkpoint(
        trajectory_dir / "best.pth",
        torch.tensor([[0.4, -0.4]], dtype=torch.float64),
        torch.tensor([0.01], dtype=torch.float64),
        epoch=99,
        lr=0.3,
    )

    checkpoint_paths = load_tracin_checkpoint_paths(trajectory_dir)

    assert [path.name for path in checkpoint_paths] == ["epoch_002.pth", "epoch_010.pth"]


def test_tracin_score_matches_manual_checkpoint_accumulation(tmp_path: Path):
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

    checkpoint_paths = load_tracin_checkpoint_paths(trajectory_dir)
    model = torch.nn.Linear(2, 1, bias=True).double()
    criterion = torch.nn.MSELoss()

    source_inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    source_targets = torch.tensor([[0.2], [0.1]], dtype=torch.float64)
    target_inputs = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    target_targets = torch.tensor([[0.3]], dtype=torch.float64)

    manual = _manual_tracin_score(
        checkpoint_paths,
        source_inputs,
        source_targets,
        target_inputs,
        target_targets,
    )
    actual = tracin_score_from_checkpoints(
        model=model,
        checkpoint_paths=checkpoint_paths,
        source_inputs=source_inputs,
        source_targets=source_targets,
        target_inputs=target_inputs,
        target_targets=target_targets,
        criterion=criterion,
        device="cpu",
    )

    assert abs(actual - manual) < 1e-12


def test_tracin_class_compute_scores_is_deterministic(tmp_path: Path):
    trajectory_dir = tmp_path / "traj"
    _save_linear_checkpoint(
        trajectory_dir / "epoch_001.pth",
        torch.tensor([[0.15, -0.05]], dtype=torch.float64),
        torch.tensor([0.01], dtype=torch.float64),
        epoch=1,
        lr=0.2,
    )

    tracin = TracIn()
    model = torch.nn.Linear(2, 1, bias=True).double()
    criterion = torch.nn.MSELoss()

    source_inputs = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    source_targets = torch.tensor([[0.1]], dtype=torch.float64)
    target_inputs = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    target_targets = torch.tensor([[0.2]], dtype=torch.float64)

    score_a = tracin.compute_scores(
        model=model,
        trajectory_dir=trajectory_dir,
        source_inputs=source_inputs,
        source_targets=source_targets,
        target_inputs=target_inputs,
        target_targets=target_targets,
        criterion=criterion,
        device="cpu",
    )
    score_b = tracin.compute_scores(
        model=model,
        trajectory_dir=trajectory_dir,
        source_inputs=source_inputs,
        source_targets=source_targets,
        target_inputs=target_inputs,
        target_targets=target_targets,
        criterion=criterion,
        device="cpu",
    )

    assert score_a == score_b


def test_tracin_score_restores_model_state_after_checkpoint_scan(tmp_path: Path):
    trajectory_dir = tmp_path / "traj"
    _save_linear_checkpoint(
        trajectory_dir / "epoch_001.pth",
        torch.tensor([[0.15, -0.05]], dtype=torch.float64),
        torch.tensor([0.01], dtype=torch.float64),
        epoch=1,
        lr=0.2,
    )

    model = torch.nn.Linear(2, 1, bias=True).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.9, 0.8]], dtype=torch.float64))
        model.bias.copy_(torch.tensor([0.7], dtype=torch.float64))
    original = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    _ = tracin_score_from_checkpoints(
        model=model,
        checkpoint_paths=load_tracin_checkpoint_paths(trajectory_dir),
        source_inputs=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        source_targets=torch.tensor([[0.1]], dtype=torch.float64),
        target_inputs=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        target_targets=torch.tensor([[0.2]], dtype=torch.float64),
        criterion=torch.nn.MSELoss(),
        device="cpu",
    )

    restored = model.state_dict()
    for name, tensor in original.items():
        assert torch.equal(restored[name], tensor)

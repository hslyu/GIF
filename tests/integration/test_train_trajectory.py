from pathlib import Path

from scripts.train._mnist_train_common import (
    build_trajectory_dir,
    trajectory_checkpoint_path,
)


def test_build_trajectory_dir_defaults_to_checkpoint_stem():
    save_path = Path("/tmp/checkpoints/mnist_resnet34.pth")

    trajectory_dir = build_trajectory_dir(save_path)

    assert trajectory_dir == Path("/tmp/checkpoints/mnist_resnet34")


def test_build_trajectory_dir_honors_explicit_override():
    save_path = Path("/tmp/checkpoints/mnist_resnet34.pth")
    explicit_dir = Path("/tmp/custom/resnet34_traj")

    trajectory_dir = build_trajectory_dir(save_path, explicit_dir)

    assert trajectory_dir == explicit_dir


def test_trajectory_checkpoint_path_uses_zero_padded_epoch_name():
    trajectory_dir = Path("/tmp/checkpoints/mnist_resnet34")

    epoch_path = trajectory_checkpoint_path(trajectory_dir, 7)

    assert epoch_path == Path("/tmp/checkpoints/mnist_resnet34/epoch_007.pth")

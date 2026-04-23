from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _load_svhn_script_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "experiments"
        / "influence"
        / "influence_scheme_comparison_svhn.py"
    )
    spec = importlib.util.spec_from_file_location("svhn_script_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


def test_svhn_trial_uses_train_targets_for_update(monkeypatch, tmp_path):
    module = _load_svhn_script_module()

    train_inputs = torch.tensor([[11.0], [12.0], [13.0], [21.0]])
    train_targets = torch.tensor([0, 0, 0, 1])
    test_inputs = torch.tensor([[91.0], [92.0], [31.0], [41.0]])
    test_targets = torch.tensor([0, 0, 1, 1])

    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets), batch_size=2, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(test_inputs, test_targets), batch_size=2, shuffle=False
    )

    class _Bundle:
        def __init__(self):
            self.train_loader = train_loader
            self.test_loader = test_loader

    captured: dict[str, torch.Tensor | int] = {}

    def fake_run_single_method(**kwargs):
        captured["sampled_inputs"] = kwargs["sampled_inputs"].clone()
        captured["sampled_targets"] = kwargs["sampled_targets"].clone()
        captured["retained_inputs"] = kwargs["retained_inputs"].clone()
        captured["retained_targets"] = kwargs["retained_targets"].clone()
        return {
            "seed": kwargs["trial_seed"],
            "method": kwargs["method_name"],
            "status": "ok",
            "reached_target": False,
            "retain_acc": 0.0,
            "self_acc": 0.0,
            "target_step": None,
            "selected_params": 0,
            "score": 0.0,
        }

    monkeypatch.setattr(
        module,
        "create_hf_data_bundle",
        lambda *args, **kwargs: _Bundle(),
    )
    monkeypatch.setattr(module, "load_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "build_model", lambda: _TinyModel())
    monkeypatch.setattr(module, "build_trajectory_path", lambda args: None)
    monkeypatch.setattr(module, "run_single_method", fake_run_single_method)

    args = Namespace(
        data_root=tmp_path,
        batch_size=2,
        num_workers=0,
        dataset_id=None,
        checkpoint=tmp_path / "dummy.pth",
        target_label=0,
        methods=["datainf"],
        param_ratios=[0.05],
        device="cpu",
        tracin_max_checkpoints=1,
        tol=1e-5,
        mu=1.0,
        max_iter=1,
        hypeinf_max_iter=1,
        edit_scale=0.01,
        max_update_steps=1,
        gif_max_self_acc_for_selection=1.5,
        num_target_batches=10,
        num_hvp_batches=1,
    )
    retrained_baseline = {"checkpoint": str(tmp_path / "retrained.pth")}

    module.run_single_trial(
        args=args,
        trial_seed=0,
        save_dir=tmp_path,
        retrained_baseline=retrained_baseline,
    )

    sampled_values = set(captured["sampled_inputs"].view(-1).tolist())
    assert sampled_values
    assert sampled_values.issubset({11.0, 12.0, 13.0})
    assert sampled_values.isdisjoint({91.0, 92.0})
    assert set(captured["sampled_targets"].tolist()) == {0}
    retained_values = set(captured["retained_inputs"].view(-1).tolist())
    assert retained_values == {21.0}
    assert set(captured["retained_targets"].tolist()) == {1}


def test_svhn_compute_method_update_uses_sampled_retained_batch_for_datainf_and_ekfac(
    monkeypatch,
):
    module = _load_svhn_script_module()
    device = torch.device("cpu")

    model = _TinyModel()
    criterion = nn.MSELoss()
    train_inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    train_targets = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets), batch_size=2, shuffle=False
    )
    sampled_inputs = torch.tensor([[5.0]])
    sampled_targets = torch.tensor([[1.0]])
    retained_inputs = torch.tensor([[10.0], [20.0], [30.0]])
    retained_targets = torch.tensor([[0.0], [1.0], [0.0]])
    batch_inputs = torch.tensor([[20.0]])
    batch_targets = torch.tensor([[1.0]])

    monkeypatch.setattr(
        module,
        "sample_hvp_batch",
        lambda *args, **kwargs: (batch_inputs.clone(), batch_targets.clone()),
    )

    captured: dict[str, torch.Tensor] = {}

    def fake_build_total_loss(model, retained_inputs, retained_targets, criterion, device):
        captured["datainf_inputs"] = retained_inputs.clone()
        captured["datainf_targets"] = retained_targets.clone()
        return (model(retained_inputs).sum() * 0) + torch.tensor(
            1.0, dtype=torch.float32
        )

    class _FakeDataInfluence:
        def compute(self, model, total_loss, target_loss, damping):
            return torch.ones(
                sum(parameter.numel() for parameter in model.parameters()),
                dtype=torch.float32,
            )

    class _FakeEKFACInfluence:
        def compute(
            self,
            model,
            retained_inputs,
            retained_targets,
            target_inputs,
            target_targets,
            criterion,
            damping,
            device,
        ):
            captured["ekfac_inputs"] = retained_inputs.clone()
            captured["ekfac_targets"] = retained_targets.clone()
            return torch.ones(
                sum(parameter.numel() for parameter in model.parameters()),
                dtype=torch.float32,
            )

    monkeypatch.setattr(module, "build_total_loss", fake_build_total_loss)
    monkeypatch.setattr(module, "DataInfluence", _FakeDataInfluence)
    monkeypatch.setattr(module, "EKFACInfluence", _FakeEKFACInfluence)

    args = Namespace(
        batch_size=2,
        mu=1.0,
        tol=1e-5,
        max_iter=1,
        solver_power_iters=1,
        lissa_damping=1e-2,
        lissa_mu_scale=1.0,
        lissa_max_restarts=1,
        hyperinf_beta_scale=0.9,
        hypeinf_max_iter=1,
        datainf_damping=1e-6,
        ekfac_damping=1e-3,
        tracin_max_checkpoints=1,
        num_hvp_batches=1,
        p_lissa_damping=0.0,
    )

    module.compute_method_update(
        method_name="datainf",
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        sampled_inputs=sampled_inputs,
        sampled_targets=sampled_targets,
        all_target_count=1,
        retained_inputs=retained_inputs,
        retained_targets=retained_targets,
        param_ratio=None,
        args=args,
        device=device,
    )
    module.compute_method_update(
        method_name="ekfac",
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        sampled_inputs=sampled_inputs,
        sampled_targets=sampled_targets,
        all_target_count=1,
        retained_inputs=retained_inputs,
        retained_targets=retained_targets,
        param_ratio=None,
        args=args,
        device=device,
    )

    torch.testing.assert_close(captured["datainf_inputs"], batch_inputs)
    torch.testing.assert_close(captured["datainf_targets"], batch_targets)
    torch.testing.assert_close(captured["ekfac_inputs"], batch_inputs)
    torch.testing.assert_close(captured["ekfac_targets"], batch_targets)

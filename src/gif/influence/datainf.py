from __future__ import annotations

import torch

from gif.models import get_trainable_parameters
from gif.solvers import datainf_inverse_diagonal


def _compute_trainable_gradient(
    model: torch.nn.Module,
    loss: torch.Tensor,
) -> torch.Tensor:
    params = get_trainable_parameters(model)
    if not params:
        raise RuntimeError("DataInf requires trainable LoRA adapter parameters.")
    grads = torch.autograd.grad(loss, params, retain_graph=True)
    return torch.cat([gradient.contiguous().view(-1) for gradient in grads])


def datainf_update(
    model: torch.nn.Module,
    total_loss: torch.Tensor,
    target_loss: torch.Tensor,
    *,
    damping: float = 1e-6,
    return_details: bool = False,
):
    g_target = _compute_trainable_gradient(model, target_loss)
    g_total = _compute_trainable_gradient(model, total_loss)
    diagonal = g_total.pow(2)
    result = datainf_inverse_diagonal(
        rhs=g_target,
        diagonal=diagonal,
        damping=damping,
        return_details=return_details,
    )
    if return_details:
        return {
            "update": result["solution"],
            "details": result["details"],
        }
    return result


class DataInfluence:
    def compute(
        self,
        model: torch.nn.Module,
        total_loss: torch.Tensor,
        target_loss: torch.Tensor,
        *,
        damping: float = 1e-6,
        return_details: bool = False,
    ):
        return datainf_update(
            model=model,
            total_loss=total_loss,
            target_loss=target_loss,
            damping=damping,
            return_details=return_details,
        )


DataInf = DataInfluence

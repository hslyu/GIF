import torch

from gif.models import get_trainable_parameters


def compute_gradient(
    model: torch.nn.Module,
    loss: torch.Tensor,
    create_graph: bool = False,
    retain_graph: bool = True,
) -> torch.Tensor:
    params = get_trainable_parameters(model)
    if not params:
        raise RuntimeError("No trainable parameters were found.")
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=create_graph,
    )
    return torch.cat([gradient.contiguous().view(-1) for gradient in grads])


def hvp(
    model: torch.nn.Module,
    loss: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    params = get_trainable_parameters(model)
    if not params:
        raise RuntimeError("No trainable parameters were found.")
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=True,
    )
    flat_grads = torch.cat([gradient.contiguous().view(-1) for gradient in grads])
    hv = torch.autograd.grad(
        flat_grads,
        params,
        grad_outputs=v,
        retain_graph=True,
    )
    return torch.cat([gradient.contiguous().view(-1) for gradient in hv])


def compute_hessian(
    model: torch.nn.Module,
    loss: torch.Tensor,
) -> torch.Tensor:
    params = get_trainable_parameters(model)
    if not params:
        raise RuntimeError("No trainable parameters were found.")
    flat_params = torch.cat([parameter.contiguous().view(-1) for parameter in params])
    dim = flat_params.numel()
    eye = torch.eye(dim, device=flat_params.device, dtype=flat_params.dtype)
    columns = [hvp(model, loss, eye[index]) for index in range(dim)]
    return torch.stack(columns, dim=1)

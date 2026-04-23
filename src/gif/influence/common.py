import torch


def compute_gradient(
    model: torch.nn.Module,
    loss: torch.Tensor,
    create_graph: bool = False,
    retain_graph: bool = True,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        list(model.parameters()),
        retain_graph=retain_graph,
        create_graph=create_graph,
    )
    return torch.cat([gradient.contiguous().view(-1) for gradient in grads])


def hvp(
    model: torch.nn.Module,
    loss: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        list(model.parameters()),
        create_graph=True,
        retain_graph=True,
    )
    flat_grads = torch.cat([gradient.contiguous().view(-1) for gradient in grads])
    hv = torch.autograd.grad(
        flat_grads,
        list(model.parameters()),
        grad_outputs=v,
        retain_graph=True,
    )
    return torch.cat([gradient.contiguous().view(-1) for gradient in hv])


def compute_hessian(
    model: torch.nn.Module,
    loss: torch.Tensor,
) -> torch.Tensor:
    flat_params = torch.cat(
        [parameter.contiguous().view(-1) for parameter in model.parameters()]
    )
    dim = flat_params.numel()
    eye = torch.eye(dim, device=flat_params.device, dtype=flat_params.dtype)
    columns = [hvp(model, loss, eye[index]) for index in range(dim)]
    return torch.stack(columns, dim=1)

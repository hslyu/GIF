import torch


def as_index_tensor(index_list, device):
    if isinstance(index_list, torch.Tensor):
        return index_list.to(device=device, dtype=torch.long)
    return torch.as_tensor(index_list, device=device, dtype=torch.long)


def embed_subset(v_sub: torch.Tensor, index_list, full_dim: int) -> torch.Tensor:
    idx = as_index_tensor(index_list, v_sub.device)
    out = torch.zeros(full_dim, device=v_sub.device, dtype=v_sub.dtype)
    out[idx] = v_sub
    return out


def project_subset(v_full: torch.Tensor, index_list) -> torch.Tensor:
    idx = as_index_tensor(index_list, v_full.device)
    return v_full.index_select(0, idx)


_as_index_tensor = as_index_tensor
_embed_subset = embed_subset
_project_subset = project_subset

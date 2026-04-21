"""Generalized Influence Functions package."""

from .influence import (
    cg_update,
    compute_gradient,
    compute_hessian,
    datainf_update,
    freeze_influence,
    generalized_influence,
    hyperinf_update,
    hvp,
    influence,
    iphvp_fif,
    lanczos_update,
    lissa_update,
    plain_influence,
    second_influence,
)
from .influence.freezing import iphvp_fif as iphvp_FIF
from .solvers import (
    cg_inverse,
    datainf_inverse_diagonal,
    hyperinf_inverse,
    ihvp,
    iphvp,
    lanczos_inverse,
    lissa_inverse,
    p_lissa,
)
from .regularization import RegularizedLoss

__all__ = [
    "RegularizedLoss",
    "cg_inverse",
    "cg_update",
    "compute_gradient",
    "compute_hessian",
    "datainf_inverse_diagonal",
    "datainf_update",
    "freeze_influence",
    "generalized_influence",
    "hyperinf_inverse",
    "hyperinf_update",
    "hvp",
    "ihvp",
    "influence",
    "iphvp",
    "p_lissa",
    "lanczos_inverse",
    "lanczos_update",
    "lissa_inverse",
    "lissa_update",
    "iphvp_FIF",
    "iphvp_fif",
    "plain_influence",
    "second_influence",
]

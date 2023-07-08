""" Use scipy/ARPACK implicitly restarted lanczos to find top k eigenthings """
from typing import Tuple
from warnings import warn

import numpy as np
import scipy.sparse.linalg as linalg
import torch
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator

from . import hessians


def maybe_fp16(vec, fp16):
    return vec.half() if fp16 else vec.float()


def lanczos(
    loss: torch.Tensor,
    model: torch.nn.Module,
    num_eigenthings: int = 0,
    which: str = "LM",
    max_steps: int = 20,
    tol: float = 1e-3,
    num_lanczos_vectors: int = -1,
    init_vec: np.ndarray = np.empty(0),
    use_gpu: bool = False,
    fp16: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use the scipy.sparse.linalg.eigsh hook to the ARPACK lanczos algorithm
    to find the top k eigenvalues/eigenvectors.

    Please see scipy documentation for details on specific parameters
    such as 'which'.

    Parameters
    -------------
    loss (torch.Tensor): Evaluated loss from the model
    model (torch.nn.Module): Model where the Hessian is computed
    num_eigenthings : int
        number of eigenvalue/eigenvector pairs to compute
    which : str ['LM', SM', 'LA', SA']
        L,S = largest, smallest. M, A = in magnitude, algebriac
        SM = smallest in magnitude. LA = largest algebraic.
    max_steps : int
        maximum number of arnoldi updates
    tol : float
        relative accuracy of eigenvalues / stopping criterion
    num_lanczos_vectors : int
        number of lanczos vectors to compute. if None, > 2*num_eigenthings
        for stability.
    init_vec: [torch.Tensor, torch.cuda.Tensor]
        if None, use random tensor. this is the init vec for arnoldi updates.
    use_gpu: bool
        if true, use cuda tensors.
    fp16: bool
        if true, keep operator input/output in fp16 instead of fp32.

    Returns
    ----------------
    eigenvalues : np.ndarray
        array containing `num_eigenthings` eigenvalues of the operator
    eigenvectors : np.ndarray
        array containing `num_eigenthings` eigenvectors of the operator
    """

    size = int(sum(p.numel() for p in model.parameters()))
    shape = (size, size)

    if num_eigenthings == 0:
        num_eigenthings = size - 1

    if num_lanczos_vectors == -1:
        num_lanczos_vectors = min(2 * num_eigenthings, size)

    if num_lanczos_vectors < 2 * num_eigenthings:
        warn(
            "[lanczos] number of lanczos vectors should usually be > 2*num_eigenthings"
        )

    def _scipy_apply(vec):
        vec = torch.from_numpy(vec)
        vec = maybe_fp16(vec, fp16)
        if use_gpu:
            vec = vec.cuda()
        out = hessians.hvp(loss, model, vec)
        out = maybe_fp16(out, fp16)
        out = out.cpu().numpy()
        return out

    scipy_op = ScipyLinearOperator(shape, _scipy_apply)
    if init_vec == np.empty(0):
        init_vec = np.random.rand(size)

    eigenvals = linalg.eigsh(
        A=scipy_op,
        k=num_eigenthings,
        which=which,
        maxiter=max_steps,
        tol=tol,
        ncv=num_lanczos_vectors,
        return_eigenvectors=False,
    )

    return eigenvals

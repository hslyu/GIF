import torch

from gif.solvers import lissa_inverse


def test_lissa_inverse_matches_exact_dense_solution_with_damping():
    A = torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64)
    rhs = torch.tensor([1.0, -2.0], dtype=torch.float64)
    damping = 0.1

    result = lissa_inverse(
        a_times=lambda x: A @ x,
        rhs=rhs,
        damping=damping,
        mu=0.25,
        tol=1e-12,
        max_iter=200,
        max_restarts=4,
    )
    expected = torch.linalg.solve(
        A + damping * torch.eye(rhs.numel(), dtype=torch.float64),
        rhs,
    )

    torch.testing.assert_close(result, expected, atol=5e-5, rtol=1e-4)


def test_lissa_inverse_more_iterations_reduce_residual():
    A = torch.tensor([[4.0, 0.2], [0.2, 1.5]], dtype=torch.float64)
    rhs = torch.tensor([1.0, 0.5], dtype=torch.float64)

    short = lissa_inverse(
        a_times=lambda x: A @ x,
        rhs=rhs,
        mu=0.1,
        tol=1e-12,
        max_iter=3,
        max_restarts=1,
    )
    long = lissa_inverse(
        a_times=lambda x: A @ x,
        rhs=rhs,
        mu=0.1,
        tol=1e-12,
        max_iter=80,
        max_restarts=1,
    )

    short_residual = torch.linalg.norm(A @ short - rhs).item()
    long_residual = torch.linalg.norm(A @ long - rhs).item()
    assert long_residual < short_residual


def test_lissa_inverse_returns_details_for_zero_rhs():
    rhs = torch.zeros(3, dtype=torch.float64)

    result = lissa_inverse(
        a_times=lambda x: x,
        rhs=rhs,
        return_details=True,
    )

    assert torch.equal(result["solution"], rhs)
    assert result["details"]["converged"] is True

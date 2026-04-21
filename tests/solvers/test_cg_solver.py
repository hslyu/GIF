import torch

from gif.solvers import cg_inverse


def test_cg_inverse_matches_exact_dense_solution():
    A = torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64)
    rhs = torch.tensor([1.0, -2.0], dtype=torch.float64)

    result = cg_inverse(
        a_times=lambda x: A @ x,
        rhs=rhs,
        tol=1e-12,
        max_iter=20,
    )
    expected = torch.linalg.solve(A, rhs)

    torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)


def test_cg_inverse_applies_damping():
    A = torch.tensor([[3.0, 0.1], [0.1, 1.0]], dtype=torch.float64)
    rhs = torch.tensor([1.0, 2.0], dtype=torch.float64)
    damping = 0.5

    result = cg_inverse(
        a_times=lambda x: A @ x,
        rhs=rhs,
        damping=damping,
        tol=1e-12,
        max_iter=20,
    )
    expected = torch.linalg.solve(
        A + damping * torch.eye(rhs.numel(), dtype=torch.float64),
        rhs,
    )

    torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)


def test_cg_inverse_returns_details():
    A = torch.tensor([[4.0, 0.2], [0.2, 1.5]], dtype=torch.float64)
    rhs = torch.tensor([1.0, 0.5], dtype=torch.float64)

    result = cg_inverse(
        a_times=lambda x: A @ x,
        rhs=rhs,
        tol=1e-12,
        max_iter=20,
        return_details=True,
    )

    assert result["details"]["converged"] is True
    assert result["details"]["iterations"] >= 1
    assert result["details"]["residuals"][-1] < result["details"]["residuals"][0]

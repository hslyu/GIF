import torch

from gif.solvers import lanczos_inverse


def test_lanczos_inverse_matches_exact_dense_solution_with_full_rank():
    A = torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64)
    rhs = torch.tensor([1.0, -2.0], dtype=torch.float64)

    result = lanczos_inverse(
        a_times=lambda x: A @ x,
        rhs=rhs,
        rank=rhs.numel(),
        tol=1e-10,
        max_iter=20,
    )
    expected = torch.linalg.solve(A, rhs)

    torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)


def test_lanczos_inverse_rank_improves_low_rank_operator_accuracy():
    A = torch.diag(torch.tensor([5.0, 2.0, 0.2], dtype=torch.float64))
    rhs = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    oracle = torch.linalg.solve(A, rhs)

    rank_one = lanczos_inverse(
        a_times=lambda x: A @ x,
        rhs=rhs,
        rank=1,
        tol=1e-10,
        max_iter=20,
    )
    full_rank = lanczos_inverse(
        a_times=lambda x: A @ x,
        rhs=rhs,
        rank=rhs.numel(),
        tol=1e-10,
        max_iter=20,
    )

    rank_one_error = torch.linalg.norm(rank_one - oracle).item()
    full_rank_error = torch.linalg.norm(full_rank - oracle).item()
    assert full_rank_error < rank_one_error


def test_lanczos_inverse_applies_damping_and_returns_details():
    A = torch.diag(torch.tensor([1e-8, 1.0, 3.0], dtype=torch.float64))
    rhs = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)

    result = lanczos_inverse(
        a_times=lambda x: A @ x,
        rhs=rhs,
        rank=rhs.numel(),
        damping=1e-3,
        tol=1e-10,
        max_iter=20,
        return_details=True,
    )
    expected = torch.linalg.solve(
        A + 1e-3 * torch.eye(rhs.numel(), dtype=torch.float64),
        rhs,
    )

    torch.testing.assert_close(result["solution"], expected, atol=1e-10, rtol=1e-10)
    assert result["details"]["relative_residual"] < 1e-8
    assert result["details"]["used_dense_fallback"] is True

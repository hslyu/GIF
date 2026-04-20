import torch

from gif.solvers import hyperinf_inverse


def test_hyperinf_inverse_matches_exact_dense_solution():
    A = torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64)
    rhs = torch.tensor([1.0, -2.0], dtype=torch.float64)

    result = hyperinf_inverse(
        a_times=lambda x: A @ x,
        rhs=rhs,
        tol=1e-10,
        max_iter=6,
    )
    expected = torch.linalg.solve(A, rhs)

    assert torch.allclose(result, expected, atol=1e-8, rtol=1e-8)


def test_hyperinf_inverse_reduces_residual_over_iterations():
    A = torch.tensor([[4.0, 0.2], [0.2, 1.5]], dtype=torch.float64)
    rhs = torch.tensor([1.0, 0.5], dtype=torch.float64)

    details = hyperinf_inverse(
        a_times=lambda x: A @ x,
        rhs=rhs,
        tol=1e-12,
        max_iter=5,
        return_details=True,
    )["details"]

    residuals = details["residuals"]
    assert len(residuals) >= 2
    assert residuals[-1] < residuals[0]


def test_hyperinf_inverse_handles_explicit_bad_beta_without_nonfinite_output():
    A = torch.tensor([[3.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    rhs = torch.tensor([1.0, 2.0], dtype=torch.float64)

    result = hyperinf_inverse(
        a_times=lambda x: A @ x,
        rhs=rhs,
        beta=1.2,
        tol=1e-12,
        max_iter=4,
        return_details=True,
    )

    assert torch.isfinite(result["solution"]).all()
    assert len(result["details"]["residuals"]) >= 1


def test_hyperinf_inverse_returns_finite_solution_on_ill_conditioned_system():
    A = torch.tensor([[1.0, 0.0], [0.0, 1e-3]], dtype=torch.float64)
    rhs = torch.tensor([1.0, 1.0], dtype=torch.float64)

    result = hyperinf_inverse(
        a_times=lambda x: A @ x,
        rhs=rhs,
        tol=1e-8,
        max_iter=8,
        return_details=True,
    )

    assert torch.isfinite(result["solution"]).all()
    assert result["details"]["iterations"] >= 1

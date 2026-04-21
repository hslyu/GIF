import torch

from gif.solvers import datainf_inverse_diagonal


def test_datainf_inverse_diagonal_matches_exact_diagonal_solution():
    diagonal = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64)
    rhs = torch.tensor([1.0, -2.0, 4.0], dtype=torch.float64)

    result = datainf_inverse_diagonal(rhs=rhs, diagonal=diagonal, damping=0.0)
    expected = rhs / diagonal

    assert torch.allclose(result, expected)


def test_datainf_inverse_diagonal_applies_damping():
    diagonal = torch.tensor([0.0, 1.0], dtype=torch.float64)
    rhs = torch.tensor([2.0, 2.0], dtype=torch.float64)

    result = datainf_inverse_diagonal(rhs=rhs, diagonal=diagonal, damping=1.0)
    expected = torch.tensor([2.0, 1.0], dtype=torch.float64)

    assert torch.allclose(result, expected)


def test_datainf_inverse_diagonal_rejects_negative_diagonal():
    diagonal = torch.tensor([1.0, -1.0], dtype=torch.float64)
    rhs = torch.tensor([1.0, 1.0], dtype=torch.float64)

    try:
        datainf_inverse_diagonal(rhs=rhs, diagonal=diagonal)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative diagonal.")


def test_datainf_inverse_diagonal_handles_ill_conditioned_diagonal():
    diagonal = torch.tensor([1e-8, 1.0, 1e3], dtype=torch.float64)
    rhs = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

    result = datainf_inverse_diagonal(
        rhs=rhs,
        diagonal=diagonal,
        damping=1e-6,
        return_details=True,
    )

    assert torch.isfinite(result["solution"]).all()
    assert result["details"]["min_diagonal"] >= 0.0

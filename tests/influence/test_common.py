import torch

from gif.influence import (
    _embed_subset,
    _project_subset,
    compute_gradient,
    compute_hessian,
    hvp,
)


def test_compute_gradient_matches_closed_form_linear_regression():
    model = torch.nn.Linear(2, 1, bias=True).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.4]], dtype=torch.float64))
        model.bias.copy_(torch.tensor([0.1], dtype=torch.float64))

    x = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    y = torch.tensor([[0.5]], dtype=torch.float64)
    loss = torch.nn.MSELoss()(model(x), y)

    grad = compute_gradient(model, loss)
    error = (model(x) - y).squeeze(0).squeeze(0)
    expected = torch.stack(
        [2.0 * error * x[0, 0], 2.0 * error * x[0, 1], 2.0 * error]
    ).to(dtype=torch.float64)

    torch.testing.assert_close(grad, expected)


def test_hvp_matches_explicit_hessian_product():
    model = torch.nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.3, -0.2]], dtype=torch.float64))

    x = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    y = torch.tensor([[1.5]], dtype=torch.float64)
    loss = torch.nn.MSELoss()(model(x), y)
    vector = torch.tensor([0.4, -0.1], dtype=torch.float64)

    hessian = compute_hessian(model, loss)
    expected = hessian @ vector
    actual = hvp(model, loss, vector)

    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_embed_and_project_subset_round_trip():
    index_list = torch.tensor([4, 1, 3], dtype=torch.long)
    v_sub = torch.tensor([0.5, -1.5, 2.0], dtype=torch.float64)

    v_full = _embed_subset(v_sub, index_list, full_dim=6)

    expected_full = torch.tensor(
        [0.0, -1.5, 0.0, 2.0, 0.5, 0.0], dtype=torch.float64
    )
    torch.testing.assert_close(v_full, expected_full)
    torch.testing.assert_close(_project_subset(v_full, index_list), v_sub)


def test_compute_hessian_matches_closed_form_quadratic():
    model = torch.nn.Linear(2, 1, bias=True).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.4]], dtype=torch.float64))
        model.bias.copy_(torch.tensor([0.1], dtype=torch.float64))

    x = torch.tensor(
        [[1.0, 2.0], [0.5, -1.0]], dtype=torch.float64
    )
    y = torch.tensor([[0.5], [0.3]], dtype=torch.float64)
    loss = torch.nn.MSELoss()(model(x), y)

    hessian = compute_hessian(model, loss)
    x_aug = torch.cat([x, torch.ones(x.size(0), 1, dtype=torch.float64)], dim=1)
    expected = (2.0 / x.size(0)) * (x_aug.T @ x_aug)

    torch.testing.assert_close(hessian, expected, atol=1e-10, rtol=1e-10)

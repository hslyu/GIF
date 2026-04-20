import torch

from gif.regularization import RegularizedLoss


def test_regularized_loss_adds_l2_penalty():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0]]))

    criterion = torch.nn.MSELoss()
    regularized = RegularizedLoss(model, criterion, alpha=0.2)

    pred = torch.tensor([[1.0]])
    target = torch.tensor([[0.0]])

    loss = regularized(pred, target)
    expected = criterion(pred, target) + 0.2 * torch.norm(model.weight.flatten()) / 2

    assert torch.allclose(loss, expected)

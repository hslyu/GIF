#! /usr/bin/env python
import numpy as np
import torch
import torch.nn as nn

import hessians


class RegularziedMSELoss(nn.Module):
    """
    Regularized MSE loss function

    Args:
        model (nn.Module): Model to be regularized
        weight (float): Weight of the regularization term
        scale (float): Scale of the loss. Useful for makeing the spectral norm of hessian <= 1
    """

    def __init__(self, model: nn.Module, scale=1.0, alpha=0.1):
        super(RegularziedMSELoss, self).__init__()
        self.model = model
        self.scale = scale
        self.alpha = alpha

    def forward(self, pred, target):
        mseloss = nn.MSELoss()
        params = torch.cat([param.view(-1) for param in self.model.parameters()])
        return (
            mseloss(pred, target) + self.alpha * torch.norm(params) / 2
        ) * self.scale


class LinearRegression(nn.Module):
    # Define the linear regression model
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


class LinearRegressionHiddenLayer(nn.Module):
    # Define the linear regression model
    def __init__(self):
        super(LinearRegressionHiddenLayer, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)
        self.linear3 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        y_pred = self.linear3(x)
        return y_pred


def __main__():
    # Generate some dummy data
    np.random.seed(0)
    num_samples = 10000
    x = np.random.rand(num_samples, 2)
    y = 2 * x[:, 0:1] + 3 * x[:, 1:2] + 1 + 0.5 * np.random.randn(num_samples, 1)

    # Convert the data to PyTorch tensors
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    model = LinearRegression()
    # model = LinearRegressionHiddenLayer()
    # Define the loss function (mean squared error) and optimizer (Stochastic Gradient Descent)
    # criterion = nn.MSELoss()
    criterion = RegularziedMSELoss(model, scale=0.8, alpha=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Train the model
    num_epochs = 1500
    for epoch in range(num_epochs):
        # Forward pass
        y_pred = model(x_tensor)
        loss = criterion(y_pred, y_tensor)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Print the learned parameters
    # w = model.linear.weight.detach().numpy()
    # b = model.linear.bias.detach().numpy()
    # print(f"Learned parameters layer: w={w}, b={b}")

    # loss of the model for computing gradient
    y_pred = model(x_tensor[:2])
    loss = criterion(y_pred, y_tensor[:2])

    # total loss of the model
    # Todo: approximated total loss will be coded.
    y_pred = model(x_tensor)
    total_loss = criterion(y_pred, y_tensor)

    gradient = hessians.compute_gradient(loss, model)
    hessian = hessians.compute_hessian(total_loss, model)

    # print(f"{hessian=}")
    print("Eigenvalues of hessian")
    print(torch.linalg.eig(hessian).eigenvalues.real)
    print("")
    # print(torch.linalg.eig((hessian @ hessian)[:2, :2]))
    # print(f"Actual HVP: \t    {hessian @ gradient}")
    # print(f"Approximated HVP:   {hessians.hvp(total_loss, model, gradient)}")
    print(f"Actual IHVP: \t    {torch.inverse(hessian) @ gradient}")
    # print(f"Approximated IHVP:  {hessians.ihvp(total_loss, model, gradient)}")
    print(f"Influence function: {hessians.influence(loss, total_loss/3, model)/3}")
    # num_params = sum(p.numel() for p in model.parameters())
    # print(
    #     f"Influence function via PIF:  \t    {hessians.partial_influence(list(range(num_params)), loss, total_loss, model)}"
    # )

    index_list = [0, 2]
    partial_hessian = hessian[:, index_list]
    print(f"Actual PIF for index {index_list}:")
    print(
        torch.linalg.inv(partial_hessian.T @ partial_hessian)
        @ partial_hessian.T
        @ gradient
    )
    print(f"Approximated PIF for index {index_list}:")
    print(hessians.partial_influence(index_list, loss, total_loss, model, tol=1e-8))
    print("Approximated PIF (Boosted):")
    print(
        hessians.partial_influence(index_list, loss, total_loss / 2.5, model, tol=1e-8)
        / 2.5
    )


if __name__ == "__main__":
    __main__()

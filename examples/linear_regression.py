#! /usr/bin/env python
import numpy as np
import torch
import torch.nn as nn

from src import hessians, lanczos


class RegularziedMSELoss(nn.Module):
    """
    Regularized MSE loss function

    Args:
        model (nn.Module): Model to be regularized
        weight (float): Weight of the regularization term
        scale (float): Scale of the loss. Useful for makeing the spectral norm of hessian <= 1
    """

    def __init__(self, model: nn.Module, weight=0.001):
        super(RegularziedMSELoss, self).__init__()
        self.model = model
        self.weight = weight

    def forward(self, pred, target):
        mseloss = nn.MSELoss()
        params = torch.cat([param.view(-1) for param in self.model.parameters()])
        return (mseloss(pred, target) + self.weight * torch.norm(params) / 2) / (
            1 + self.weight
        )


class LinearRegression(nn.Module):
    # Define the linear regression model
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(5, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


class LinearRegressionHiddenLayer(nn.Module):
    # Define the linear regression model
    def __init__(self):
        super(LinearRegressionHiddenLayer, self).__init__()
        self.linear1 = nn.Linear(5, 5)
        self.linear2 = nn.Linear(5, 2)
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
    x = np.random.rand(num_samples, 5)
    y = (
        2 * x[:, 0:1]
        + 3 * x[:, 1:2]
        - x[:, 2:3]
        + 5 * x[:, 3:4]
        - 4 * x[:, 4:5]
        + 0.5 * np.random.randn(num_samples, 1)
        + 1
    )

    # Convert the data to PyTorch tensors
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()
    print(x_tensor.size(), y_tensor.size())

    # model = LinearRegression()
    model = LinearRegressionHiddenLayer()
    # Define the loss function (mean squared error) and optimizer (Stochastic Gradient Descent)
    # criterion = nn.MSELoss()
    criterion = RegularziedMSELoss(model, 3)
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

    gradient = hessians.compute_gradient(model, loss)
    hessian = hessians.compute_hessian(model, total_loss)

    print("-----------Inverse hessian vector product accuracy test------------\n")
    # print(torch.linalg.eig((hessian @ hessian)[:2, :2]))
    # print(f"Actual HVP: \t    {hessian @ gradient}")
    # print(f"Approximated HVP:   {hessians.hvp(total_loss, model, gradient)}")
    print(f"Actual IHVP: \t    {torch.inverse(hessian) @ gradient}")
    # print(f"Approximated IHVP:  {hessians.ihvp(total_loss, model, gradient)}")
    print(f"Influence function: {hessians.influence(model, total_loss/3, loss)/3}")
    # num_params = sum(p.numel() for p in model.parameters())
    # print(
    #     f"Influence function via PIF:  \t    {hessians.partial_influence(list(range(num_params)), loss, total_loss, model)}"
    # )

    print("-----------Partial influence function accuracy test------------\n")
    index_list = [0, 2]
    partial_hessian = hessian[:, index_list]
    print(f"Actual PIF for index {index_list}:")
    print(
        torch.linalg.inv(partial_hessian.T @ partial_hessian)
        @ partial_hessian.T
        @ gradient
    )
    print(f"Approximated PIF for index {index_list}:")
    print(hessians.partial_influence(model, total_loss, loss, index_list))

    print("-----------Lanzcos algorithm test------------\n")
    eigvals = lanczos.lanczos(model, loss, num_lanczos_vectors=100)
    num_lanczos_eigval = sum(p.numel() for p in model.parameters()) - 1
    print(
        f"Negative eigval rate by Lanczos algorithms: {np.sum(eigvals < -1e-3)/num_lanczos_eigval}"
    )
    print(f"Lanczos algorithms: \n{np.sort(eigvals)}")
    # print("Eigenvalues of hessian")
    # print(torch.sort(torch.linalg.eig(hessian).eigenvalues.real).values)


if __name__ == "__main__":
    __main__()

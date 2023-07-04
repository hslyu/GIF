import torch
import torch.nn as nn


class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 5, 7, 5, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(5, 10, 3, 2, 1),
            nn.ReLU(),
        )
        size = self.layers(torch.ones(1, 1, 28, 28)).size(3)
        self.layers.append(
            nn.Flatten(),
        )
        self.layers.append(nn.Linear(size**2 * 10, 10))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    a = TinyNet()
    num_params = sum(p.numel() for p in a.parameters() if p.requires_grad)
    print(num_params)
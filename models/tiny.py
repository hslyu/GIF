import torch
import torch.nn as nn


class TinyNet(nn.Module):
    def __init__(self, **kwargs):
        super(TinyNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
        )
        size = self.layers(torch.ones(1, 1, 28, 28)).size(3)
        self.layers.append(
            nn.Flatten(),
        )
        self.layers.append(nn.Linear(size**2 * 256, 10))
        # self.layers.append(nn.Linear(100, 10))

    def forward(self, x):
        # for layer in self.layers:
        #     x = layer(x)
        x = self.layers(x)
        return x


if __name__ == "__main__":
    a = TinyNet()
    num_params = sum(p.numel() for p in a.parameters() if p.requires_grad)
    print(num_params)

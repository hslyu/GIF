"""TinyNet in PyTorch."""
import torch.nn as nn
import torch.nn.functional as F


class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, out):
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

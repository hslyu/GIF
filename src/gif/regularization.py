import torch
from torch import nn


class RegularizedLoss(nn.Module):
    def __init__(self, net, criterion, alpha=1e-4):
        super(RegularizedLoss, self).__init__()
        self.net = net
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        params = torch.cat([param.view(-1) for param in self.net.parameters()])
        loss += self.alpha * torch.norm(params) / 2
        return loss

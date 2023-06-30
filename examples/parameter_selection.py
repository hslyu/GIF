#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time

import torch
import torch.backends.cudnn as cudnn
from torch import nn

from dataloader import mnist
from models import FullyConnectedNet
from src import lanczos, parameter_selection, regularization

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    net = FullyConnectedNet(28 * 28, 8, 10, 3, 0.1).to(device)
    net_name = net.__class__.__name__
    print(
        f"==> Building {net_name} finished. "
        + f"\n    Number of parameters: {sum(p.numel() for p in net.parameters())}"
    )

    if device == "cuda":
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    # Data
    print("==> Preparing data..")
    batch_size = 512
    num_workers = 2

    data_loader = mnist.MNISTDataLoader(batch_size, num_workers, flatten=True)
    _, _, test_loader = data_loader.get_data_loaders()

    # Make hooks
    parameter_parser = parameter_selection.TopNActivations(net, 100)
    parameter_parser.register_hook()

    # One batch of train data
    data, target = next(iter(test_loader))
    loss = criterion(net(data.to(device)), target.to(device))


if __name__ == "__main__":
    main()

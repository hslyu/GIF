#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
from dataclasses import dataclass

import torch
import torch.backends.cudnn as cudnn
from torch import nn

from dataloader import mnist
from models import FullyConnectedNet, TinyNet
from src import hessians, parameter_selection, utils

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_net(net, path):
    assert os.path.isfile(path), "Error: no checkpoint file found!"
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint["net"])
    train_epoch = checkpoint["epoch"]
    return net, train_epoch


def forward(net, dataloader, criterion):
    net.train()
    train_loss = 0
    # Experimental lines for just doing some tests
    # count = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss

        utils.progress_bar(batch_idx, len(dataloader), "")
        # Experimental lines for just doing some tests
        # count += 1
        # if count == 5:
        #     return train_loss

    train_loss /= len(dataloader)
    return train_loss


def main():
    torch.manual_seed(0)
    net = FullyConnectedNet(28 * 28, 8, 10, 3, 0.1).to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    flatten = True
    net, train_epoch = load_net(
        net,
        "/home/hslyu/research/PIF/checkpoints/Figure_3/FullyConnectedNet/cross_entropy/ckpt_0.0.pth",
    )

    # For just doing some exps
    # net = FullyConnectedNet(28 * 28, 600, 10, 200, 0.1).to(device)
    # train_epoch = 50
    net = TinyNet().to(device)
    train_epoch = 50
    flatten = False

    net.eval()
    net_name = net.__class__.__name__
    num_param = sum(p.numel() for p in net.parameters())
    print(
        f"==> Building {net_name} finished. "
        + f"\n    Number of parameters: {num_param}"
    )

    criterion = nn.CrossEntropyLoss()

    # Data
    print("==> Preparing data..")
    batch_size = 512
    num_workers = 2

    data_loader = mnist.MNISTDataLoader(batch_size, num_workers, flatten=flatten)
    train_loader, val_loader, _ = data_loader.get_data_loaders()

    print("==> Computing total loss..")
    total_loss = forward(net, train_loader, criterion)
    print(total_loss)

    print("==> Registering hooks..")
    # Make hooks
    net_parser = parameter_selection.TopNActivations(net, int(num_param * 0.5))
    net_parser.register_hook()

    print("==> Computing influence..")
    # One batch of train data
    for _ in range(train_epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            print("batch_idx: ", batch_idx)
            net_parser.initialize_neurons()

            idx = target == 8
            data = data[idx]
            target = target[idx]

            target_loss = criterion(net(data.to(device)), target.to(device))
            index_list = net_parser.get_parameters()
            influence = hessians.partial_influence(
                index_list, target_loss, total_loss, net
            )

    net_parser.remove_hooks()


if __name__ == "__main__":
    main()

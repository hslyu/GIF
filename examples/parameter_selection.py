#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn

from dataloader import mnist
from models import FullyConnectedNet, ResNet18, TinyNet
from src import hessians, selection, utils

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_net(net, path):
    assert os.path.isfile(path), "Error: no checkpoint file found!"
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint["net"])
    return net


def save_net(net, path):
    dir, filename = os.path.split(path)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    state = {
        "net": net.state_dict(),
    }
    torch.save(state, path)


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
        # return train_loss

    train_loss /= len(dataloader)
    return train_loss


def test(net, dataloader, criterion, label, include):
    net_loss = 0
    correct = 0
    total = 0
    for _, (inputs, targets) in enumerate(dataloader):
        if include:
            idx = targets == label
        else:
            idx = targets != label
        inputs = inputs[idx]
        targets = targets[idx]
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        net_loss += loss

        total += targets.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

    accuracy = correct / total * 100
    net_loss /= len(dataloader)
    return net_loss, accuracy


def main():
    torch.manual_seed(0)
    net = FullyConnectedNet(28 * 28, 20, 10, 3, 0.1).to(device)
    flatten = True

    # net = ResNet18().to(device)
    # flatten = False

    if device == "cuda":
        cudnn.benchmark = True

    net_path = f"/home/hslyu/research/PIF/checkpoints/Figure_3/{net.__class__.__name__}/cross_entropy/ckpt_0.0.pth"
    net = load_net(net, net_path)

    # For just doing some exps
    # net = FullyConnectedNet(28 * 28, 600, 10, 200, 0.1).to(device)
    # net = TinyNet().to(device)
    # flatten = False

    net.eval()
    net_name = net.__class__.__name__
    num_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
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

    print("==> Registering hooks..")
    # Make hooks
    percentage = 0.3
    # net_parser = selection.TopNActivations(net, int(num_param * percentage))
    # net_parser = selection.TopNGradients(net, int(num_param * percentage))
    # net_parser = selection.RandomSelection(net, int(num_param * percentage))
    net_parser = selection.Threshold(net, int(num_param * percentage), 1)
    net_parser.register_hooks()

    print("==> Computing influence..")
    # # One batch of train data
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     print("batch_idx: ", batch_idx)
    #     net_parser.initialize_neurons()
    #
    #     idx = target == 8
    #     data = data[idx]
    #     target = target[idx]
    #
    #     target_loss = (
    #         criterion(net(data.to(device)), target.to(device))
    #         * len(data)
    #         / len(train_loader.dataset)
    #     )
    #     target_loss.backward(retain_graph=True)
    #     index_list = net_parser.get_parameters()
    #     influence = hessians.partial_influence(
    #         index_list, target_loss, total_loss, net, tol=1e-4, step=0.1
    #     )
    #     utils.update_network(net, influence, index_list)
    #     ratio = len(train_loader.dataset) / (len(train_loader.dataset) - len(data))
    #     # Update total loss
    #     total_loss = total_loss * ratio - target_loss * (1 - ratio)
    #

    data = list()
    target = list()
    for batch_idx, (data_raw, target_raw) in enumerate(train_loader):
        net_parser.initialize_neurons()
        idx = target_raw == 8
        data_raw = data_raw[idx]
        target_raw = target_raw[idx]
        data.append(data_raw)
        target.append(target_raw)
    data = torch.cat(data)
    target = torch.cat(target)

    net_parser.initialize_neurons()
    target_loss = (
        criterion(net(data.to(device)), target.to(device))
        * len(data)
        / len(train_loader.dataset)
    )
    target_loss.backward(retain_graph=True)
    data_ratio = len(train_loader.dataset) / (len(train_loader.dataset) - len(data))
    newton_loss = total_loss * data_ratio - target_loss * (1 - data_ratio)

    index_list = net_parser.get_parameters()
    influence = hessians.partial_influence(
        index_list, target_loss, newton_loss, net, tol=1e-4
    )
    utils.update_network(net, influence * 10, index_list)
    net_parser.remove_hooks()
    save_net(
        net, f"checkpoints/Figure_3/PIF/{net_name}/{net_parser.__class__.__name__}.pth"
    )

    net = FullyConnectedNet(28 * 28, 20, 10, 3, 0.1).to(device)

    net_parser = selection.Threshold(net, int(num_param * percentage), 1)
    _, _, test_loader = data_loader.get_data_loaders()
    net_path = (
        f"checkpoints/Figure_3/PIF/{net_name}/{net_parser.__class__.__name__}.pth"
    )
    net = load_net(net, net_path)

    self_loss, self_acc = test(net, test_loader, criterion, 8, True)
    exclusive_loss, exclusive_acc = test(net, test_loader, criterion, 8, False)
    print(
        f"{net_parser.__class__.__name__}- Self: {self_loss:.4f} {self_acc:.2f}% | exclusive loss: {exclusive_loss:.4f}, {exclusive_acc:.2f}%"
    )
    print("")


if __name__ == "__main__":
    main()

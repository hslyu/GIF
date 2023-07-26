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

from dataloader import cifar10
from models import DenseNet121
from src import freeze_influence, hessians, selection, utils

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


def forward(net: nn.Module, dataloader, criterion, num_sample_batch: int = -1):
    net_loss = 0
    num_sample_batch = len(dataloader) if num_sample_batch == -1 else num_sample_batch
    sample_indices = np.random.choice(
        len(dataloader), size=num_sample_batch, replace=False
    )
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if batch_idx in sample_indices:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            net_loss += loss

    net_loss /= num_sample_batch
    return net_loss


def test(net, dataloader, criterion, label, include):
    with torch.no_grad():
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


def get_full_param_list(net):
    """
    Return a list of parameter indices in flatten network.
    Warning: this function only provides indices of params when the param i) has requires_grad=True and 2) belongs to nn.Linear or nn.Conv2d
    """

    index_list = np.array([], dtype=int)
    start_index = 0
    for module in net.modules():
        if not list(module.children()) == []:
            continue

        num_param = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module_index_list = np.arange(num_param, dtype=int) + start_index
            index_list = np.append(index_list, module_index_list)

        start_index += num_param

    return index_list


def projected_influence(net, index_list, total_loss, target_loss, tol, step):
    full_param_list = get_full_param_list(net)
    influence = hessians.partial_influence(
        full_param_list, target_loss, total_loss, net, tol=tol, step=step
    )
    idx = np.isin(full_param_list, index_list)
    return influence[idx], full_param_list[idx]


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    net = DenseNet121().to(device)
    flatten = False
    net_name = net.__class__.__name__

    if device == "cuda":
        cudnn.benchmark = True

    net_path = f"checkpoints/Figure_4/{net_name}/cross_entropy/ckpt_0.0.pth"
    net = load_net(net, net_path)

    net.eval()
    num_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(
        f"==> Building {net_name} finished. "
        + f"\n    Number of parameters: {num_param}"
    )

    criterion = nn.CrossEntropyLoss()

    # Data
    print("==> Preparing data..")
    batch_size = 128
    num_workers = 12
    num_sample_batch = 1
    num_target_sample = 50

    data_loader = cifar10.CIFAR10DataLoader(batch_size, num_workers, flatten=flatten)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    print("==> Computing total loss..")
    inputs_list = list()
    targets_list = list()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx < num_sample_batch:
            inputs_list.append(inputs)
            targets_list.append(targets)
        else:
            break

    total_loss = 0
    for inputs, targets in zip(inputs_list, targets_list):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss

    total_loss /= num_sample_batch

    print("==> Registering hooks..")
    # Make hooks
    percentage = 0.1
    net_parser = selection.TopNActivations(net, percentage)
    # net_parser = selection.TopNGradients(net, int(num_param * percentage))
    # net_parser = selection.RandomSelection(net, int(num_param * percentage))
    # net_parser = selection.Threshold(net, int(num_param * percentage), 1)
    net_parser.register_hooks()

    print("==> Computing influence..")
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
    sample_idx = np.random.choice(len(data), num_target_sample, replace=False)
    sample_data = data[sample_idx]
    sample_target = target[sample_idx]

    target_loss = (
        criterion(net(sample_data.to(device)), sample_target.to(device))
        * len(data)
        / len(train_loader.dataset)
    )
    target_loss.backward(retain_graph=True)
    data_ratio = len(train_loader.dataset) / (len(train_loader.dataset) - len(data))
    newton_loss = total_loss * data_ratio - target_loss * (1 - data_ratio)

    index_list = net_parser.get_parameters()

    tol = 9e-8
    influence = hessians.partial_influence(
        index_list, target_loss, newton_loss, net, tol=tol, step=0.5
    )

    # influence, index_list = projected_influence(
    #     net, index_list, newton_loss, target_loss, tol=tol, step=0.5
    # )

    # influence = freeze_influence.freeze_influence(
    #     index_list, target_loss, newton_loss, net, tol=tol, step=3
    # )

    utils.update_network(net, influence, index_list)
    net_parser.remove_hooks()
    save_path = (
        f"checkpoints/Figure_3/PIF/{net_name}/{net_parser.__class__.__name__}.pth"
    )
    save_net(net, save_path)

    net = ResNet18(1).to(device)
    net = load_net(net, save_path)

    # loss, acc = test(net, test_loader, criterion, 11, False)
    # print(f"{net_parser.__class__.__name__} original - {loss:.4f}, {acc:.2f}%")
    self_loss, self_acc = test(net, test_loader, criterion, 8, True)
    # print(f"{net_parser.__class__.__name__} - Self: {self_loss:.4f} {self_acc:.2f}%")
    exclusive_loss, exclusive_acc = test(net, test_loader, criterion, 8, False)
    print(
        f"{net_parser.__class__.__name__} Self: {self_loss:.4f} {self_acc:.2f}% | Exclusive loss: {exclusive_loss:.4f}, {exclusive_acc:.2f}%"
    )


if __name__ == "__main__":
    main()

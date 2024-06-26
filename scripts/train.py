#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import config
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn

from dataloader import cifar10, mnist, svhn
from models import VGG, VGG11, FullyConnectedNet, ResNet18, ShuffleNetV2
from src import regularization, utils

device = "cuda" if torch.cuda.is_available() else "cpu"


def _correct_fn(predicted: torch.Tensor, targets: torch.Tensor):
    if targets.dim() == 1:
        return predicted.eq(targets).sum().item()
    elif targets.dim() == 2:
        _, targets_decoded = targets.max(1)
        return predicted.eq(targets_decoded).sum().item()
    else:
        return 0


# Training
def train(net, dataloader, optimizer, criterion, exclusive_label=None):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if exclusive_label is not None:
            idx = targets != exclusive_label
            inputs, targets = inputs[idx], targets[idx]
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        correct += _correct_fn(predicted, targets)

        utils.progress_bar(
            batch_idx,
            len(dataloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )


def test(net, dataloader, criterion, epoch, configs, exclusive_label=None):
    global best_loss
    global count

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if exclusive_label is not None:
                idx = targets != exclusive_label
                inputs, targets = inputs[idx], targets[idx]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()
            correct += _correct_fn(predicted, targets)

            utils.progress_bar(
                batch_idx,
                len(dataloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )
    test_loss = test_loss / len(dataloader)
    if test_loss < best_loss:
        dirname = f"checkpoints/{configs.path}/{configs.net_name}/{configs.criterion}/"
        filename = (
            f"ckpt_{configs.alpha}.pth"
            if configs.exclusive_label is None
            else f"ckpt_{configs.alpha}_{configs.exclusive_label}.pth"
        )
        print(f"Saving the model: {test_loss:.3f} < {best_loss:.3f}")
        print(dirname + filename)
        state = {
            "net": net.state_dict(),
            "loss": test_loss,
            "epoch": epoch,
            "alpha": configs.alpha,
            "criterion": configs.criterion,
            "count": count,
        }
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        torch.save(
            state,
            dirname + filename,
        )
        best_loss = test_loss
        count = 0
    else:
        count += 1

    if count >= configs.early_stop:
        print("Early Stopping")
        return True

    return False


def main():
    torch.manual_seed(0)
    configs = config.tab2_configs(net_name="ResNet18", data="MNIST")
    configs = config.tab2_configs(net_name="ShuffleNetV2", data="SVHN")
    configs = config.tab2_configs(net_name="VGG11", data="CIFAR10")
    configs = config.tab2_configs(
        net_name="VGG11", data="CIFAR10", retrained=True, exclusive_label=1
    )

    # Network configuration
    print("==> Building Model..")
    if configs.net_name == "ResNet18":
        net = ResNet18(1).to(device)
    elif configs.net_name == "VGG11":
        net = VGG11().to(device)
    elif configs.net_name == "ShuffleNetV2":
        net = ShuffleNetV2().to(device)
    else:
        net = None
        print("Error: invalid network name")
        return

    if configs.retrained:
        configs.net_name += "_retrained"

    print(
        f"==> Building {configs.net_name} finished. "
        + f"\n    Number of parameters: {sum(p.numel() for p in net.parameters())}"
    )

    if device == "cuda":
        cudnn.benchmark = True

    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    global best_loss
    global count

    if configs.resume:
        dirname = f"checkpoints/{configs.path}/{configs.net_name}/{configs.criterion}/"
        filename = (
            f"ckpt_{configs.alpha}.pth"
            if configs.exclusive_label is None
            else f"ckpt_{configs.alpha}_{configs.exclusive_label}.pth"
        )
        assert os.path.isfile(dirname + filename), "Error: no checkpoint file found!"
        checkpoint = torch.load(
            dirname + filename,
        )
        net.load_state_dict(checkpoint["net"])
        best_loss = checkpoint["loss"]
        start_epoch = checkpoint["epoch"]
        count = checkpoint["count"]
        assert (
            checkpoint["alpha"] == configs.alpha
        ), "Error: alpha is not equal to checkpoint value!"
        assert (
            checkpoint["criterion"] == configs.criterion
        ), "Error: loss is not equal to checkpoint value!"
    else:
        best_loss = 1e9
        start_epoch = 0
        count = 0

    if configs.criterion == "cross_entropy":
        criterion = regularization.RegularizedLoss(
            net, nn.CrossEntropyLoss(), configs.alpha
        )
    else:
        criterion = regularization.RegularizedLoss(net, nn.MSELoss(), configs.alpha)
    print(
        f"==> Current criterion: {criterion.__class__.__name__} with {configs.criterion} and alpha={configs.alpha}"
    )
    optimizer = optim.SGD(
        net.parameters(), lr=configs.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    if isinstance(net, VGG):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Data
    print("==> Preparing data..")
    batch_size = configs.batch_size
    num_workers = 12

    if configs.data == "CIFAR10":
        data_loader = cifar10.CIFAR10DataLoader(batch_size, num_workers)
    elif configs.data == "MNIST":
        data_loader = mnist.MNISTDataLoader(batch_size, num_workers)
    elif configs.data == "SVHN":
        data_loader = svhn.SVHNDataLoader(batch_size, num_workers)
    else:
        print("Error: invalid dataset name")
        return
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    for epoch in range(start_epoch, start_epoch + configs.num_epoch):
        print("\nEpoch: %d" % epoch)
        train(net, train_loader, optimizer, criterion, configs.exclusive_label)
        train(net, val_loader, optimizer, criterion, configs.exclusive_label)
        early_stopping_flag = test(
            net,
            test_loader,
            criterion,
            epoch,
            configs,
            exclusive_label=configs.exclusive_label,
        )
        scheduler.step()

        if early_stopping_flag:
            print("Early Stopping")
            break


if __name__ == "__main__":
    main()

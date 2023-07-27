#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
from dataclasses import dataclass

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn

from dataloader import cifar10, mnist
from models import DenseNet121
from src import regularization, utils

# from src import hessians

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    lr: float = 0.1
    num_epoch: int = 200
    alpha: float = 0.0
    # network: str = "ResNet18"
    network: str = "DenseNet"
    data: str = "CIFAR10"
    criterion: str = "cross_entropy"
    resume: bool = False
    path: str = "Figure_4"
    early_stop: int = 30
    manual_seed: int = 0


def _correct_fn(predicted: torch.Tensor, targets: torch.Tensor):
    if targets.dim() == 1:
        return predicted.eq(targets).sum().item()
    elif targets.dim() == 2:
        _, targets_decoded = targets.max(1)
        return predicted.eq(targets_decoded).sum().item()
    else:
        return 0


# Training
def train(net, dataloader, optimizer, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
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


def test(net, net_name, dataloader, criterion, epoch, configs):
    global best_loss
    global count

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
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
        print(f"Saving the model: {test_loss:.3f} < {best_loss:.3f}")
        print(
            f"checkpoints/{configs.path}/{net_name}/{configs.criterion}/ckpt_{configs.alpha}.pth"
        )
        state = {
            "net": net.state_dict(),
            "loss": test_loss,
            "epoch": epoch,
            "alpha": configs.alpha,
            "criterion": configs.criterion,
            "count": count,
        }
        if not os.path.isdir(
            f"checkpoints/{configs.path}/{net_name}/{configs.criterion}"
        ):
            os.makedirs(f"checkpoints/{configs.path}/{net_name}/{configs.criterion}/")
        torch.save(
            state,
            f"checkpoints/{configs.path}/{net_name}/{configs.criterion}/ckpt_{configs.alpha}.pth",
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
    configs = Config()
    torch.manual_seed(0)

    # Network configuration
    print("==> Building Model..")
    net = DenseNet121().to(device)
    flatten = False
    net_name = net.__class__.__name__
    print(
        f"==> Building {net_name} finished. "
        + f"\n    Number of parameters: {sum(p.numel() for p in net.parameters())}"
    )

    if device == "cuda":
        cudnn.benchmark = True

    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    global best_loss
    global count

    if configs.resume:
        assert os.path.isfile(
            f"checkpoints/{configs.path}/{net_name}/{configs.criterion}/ckpt_{configs.alpha}.pth"
        ), "Error: no checkpoint file found!"
        checkpoint = torch.load(
            f"checkpoints/{configs.path}/{net_name}/{configs.criterion}/ckpt_{configs.alpha}.pth"
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
        one_hot = False
    else:
        criterion = regularization.RegularizedLoss(net, nn.MSELoss(), configs.alpha)
        one_hot = True
    print(
        f"==> Current criterion: {criterion.__class__.__name__} with {configs.criterion} and alpha={configs.alpha}"
    )
    optimizer = optim.SGD(
        net.parameters(), lr=configs.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Data
    print("==> Preparing data..")
    batch_size = 256
    num_workers = 16

    if configs.data == "CIFAR10":
        data_loader = cifar10.CIFAR10DataLoader(batch_size, num_workers, one_hot)
    else:
        data_loader = mnist.MNISTDataLoader(batch_size, num_workers, one_hot, flatten)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    for epoch in range(start_epoch, start_epoch + configs.num_epoch):
        print("\nEpoch: %d" % epoch)
        train(net, train_loader, optimizer, criterion)
        train(net, val_loader, optimizer, criterion)
        early_stopping_flag = test(
            net, net_name, test_loader, criterion, epoch, configs
        )
        scheduler.step()

        if early_stopping_flag:
            print("Early Stopping")
            break


if __name__ == "__main__":
    main()
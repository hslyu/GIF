#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn

from dataloader import cifar10
from models import VGG16  # ResNet50, ShuffleNetV2
from src import utils

# from src import hessians

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )
    return parser.parse_args()


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
        correct += predicted.eq(targets).sum().item()

        utils.progress_bar(
            batch_idx,
            len(dataloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )


def test(net, dataloader, criterion, epoch, best_acc):
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
            correct += predicted.eq(targets).sum().item()

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

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        print(f"Saving the model: {acc} > {best_acc}")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoints"):
            os.mkdir("checkpoints")
        torch.save(state, "checkpoints/ckpt.pth")
        best_acc = acc


def main():
    args = get_args()

    batch_size = 512
    num_workers = 12

    # Data
    print("==> Preparing data..")
    data_loader = cifar10.CIFAR10DataLoader(batch_size, num_workers)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    print("==> Building Model..")
    # net_VGG16 = VGG16().to(device)
    # net_ResNet50 = ResNet50().to(device)
    # net_ShuffleNetV2 = ShuffleNetV2().to(device)
    net = VGG16().to(device)

    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        # net_ResNet50 = torch.nn.DataParallel(net_ResNet50)
        # net_ShuffleNetV2 = torch.nn.DataParallel(net_ShuffleNetV2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    global best_acc

    if args.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir("checkpoints"), "Error: no checkpoint directory found!"
        checkpoint = torch.load("checkpoints/ckpt.pth")
        net.load_state_dict(checkpoint["net"])
        best_acc = checkpoint["acc"]
        start_epoch = checkpoint["epoch"]
    else:
        best_acc = 0
        start_epoch = 0

    for epoch in range(start_epoch, start_epoch + 50):
        print("\nEpoch: %d" % epoch)
        train(net, train_loader, optimizer, criterion)
        train(net, val_loader, optimizer, criterion)
        test(net, test_loader, criterion, epoch, best_acc)
        scheduler.step()


if __name__ == "__main__":
    main()

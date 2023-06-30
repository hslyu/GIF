#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn

from dataloader import cifar10, mnist
from models import FullyConnectedNet, LeNet  # ResNet50, ShuffleNetV2
from src import regularization, utils

# from src import hessians

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--path", default=1, type=int, help="experiment number")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--num_epoch", default=50, type=int, help="learning rate")
    parser.add_argument("--alpha", type=float, help="Regularlization weight")
    parser.add_argument(
        "--network",
        default="LeNet",
        choices=["LeNet", "FCN"],
        type=str,
        help="Model to be tested",
    )
    parser.add_argument(
        "--data",
        default="CIFAR10",
        choices=["CIFAR10", "MNIST"],
        type=str,
        help="Dataset to be used",
    )
    parser.add_argument(
        "--criterion",
        default="cross_entropy",
        choices=["cross_entropy", "mse"],
        type=str,
        help="Loss criterion function",
    )
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )
    return parser.parse_args()


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


def test(net, net_name, dataloader, criterion, epoch, args):
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

    if test_loss < best_loss:
        dir_path = f"checkpoints/Figure_2/{args.path}/{net_name}/{args.criterion}/"
        file_name = f"ckpt_{args.alpha}.pth"
        print(f"Saving the model: {test_loss:.2f} < {best_loss:.2f}")
        print(dir_path + file_name)
        state = {
            "net": net.state_dict(),
            "loss": test_loss,
            "epoch": epoch,
            "alpha": args.alpha,
            "criterion": args.criterion,
            "count": count,
        }
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        torch.save(
            state,
            dir_path + file_name,
        )
        best_loss = test_loss
        count = 0
    else:
        count += 1

    if count >= 5:
        print("Early Stopping")
        return True

    return False


def main():
    args = get_args()
    torch.manual_seed(int(args.path))

    # Network configuration
    print("==> Building Model..")
    if args.network == "LeNet":
        net = LeNet().to(device)
        flatten = False
    else:
        net = FullyConnectedNet(28 * 28, 8, 10, 3, 0.1).to(device)
        flatten = True
    net_name = net.__class__.__name__
    print(
        f"==> Building {net_name} finished. "
        + f"\n    Number of parameters: {sum(p.numel() for p in net.parameters())}"
    )

    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    global best_loss
    global count

    if args.resume:
        dir_path = f"checkpoints/{args.path}/{net_name}/{args.criterion}/"
        file_name = f"ckpt_{args.alpha}.pth"
        assert os.path.isfile(dir_path + file_name), "Error: no checkpoint file found!"
        checkpoint = torch.load(dir_path + file_name)
        net.load_state_dict(checkpoint["net"])
        best_loss = checkpoint["loss"]
        start_epoch = checkpoint["epoch"]
        count = checkpoint["count"]
        assert (
            checkpoint["alpha"] == args.alpha
        ), "Error: alpha is not equal to checkpoint value!"
        assert (
            checkpoint["criterion"] == args.criterion
        ), "Error: loss is not equal to checkpoint value!"
    else:
        best_loss = 1e9
        start_epoch = 0
        count = 0

    if args.criterion == "cross_entropy":
        criterion = regularization.RegularizedLoss(
            net, nn.CrossEntropyLoss(), args.alpha
        )
        one_hot = False
    else:
        criterion = regularization.RegularizedLoss(net, nn.MSELoss(), args.alpha)
        one_hot = True
    print(
        f"==> Current criterion: {criterion.__class__.__name__} with {args.criterion} and alpha={args.alpha}"
    )
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Data
    print("==> Preparing data..")
    batch_size = 512
    num_workers = 4

    if args.data == "CIFAR10":
        data_loader = cifar10.CIFAR10DataLoader(batch_size, num_workers, one_hot)
    else:
        data_loader = mnist.MNISTDataLoader(batch_size, num_workers, one_hot, flatten)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    for epoch in range(start_epoch, start_epoch + args.num_epoch):
        print("\nEpoch: %d" % epoch)
        train(net, train_loader, optimizer, criterion)
        train(net, val_loader, optimizer, criterion)
        early_stopping_flag = test(net, net_name, test_loader, criterion, epoch, args)
        scheduler.step()

        if early_stopping_flag:
            print("Early Stopping")
            break


if __name__ == "__main__":
    main()

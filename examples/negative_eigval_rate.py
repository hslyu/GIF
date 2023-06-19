#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn

from dataloader import cifar10, mnist
from models import VGG16, FullyConnectedNet, LeNet  # ResNet50, ShuffleNetV2
from src import hessians, lanczos, regularization, utils

# from src import hessians

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument(
        "--network",
        default="LeNet",
        choices=["LeNet", "FCN"],
        type=str,
        help="Model to be tested",
    )
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
    parser.add_argument(
        "--num_epoch", default=50, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--data",
        default="CIFAR10",
        choices=["CIFAR10", "MNIST"],
        type=str,
        help="Dataset to be used",
    )
    parser.add_argument(
        "--loss",
        default="cross_entropy",
        choices=["cross_entropy", "mse"],
        type=str,
        help="Loss function",
    )
    parser.add_argument(
        "--num_eigval", default=300, type=int, help="Number of eigenvalues to compute"
    )
    parser.add_argument(
        "--alpha", default=0.0, type=float, help="Regularization weight"
    )
    parser.add_argument(
        "--resume", "-r", action="store_true", help="Resume from checkpoint"
    )
    return parser.parse_args()


def main():
    args = get_args()

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
    assert os.path.isfile(
        f"checkpoints/{net_name}/{args.loss}/ckpt_{args.alpha}.pth"
    ), "Error: no checkpoint file found!"
    checkpoint = torch.load(f"checkpoints/{net_name}/{args.loss}/ckpt_{args.alpha}.pth")
    net.load_state_dict(checkpoint["net"])
    assert (
        checkpoint["alpha"] == args.alpha
    ), "Error: alpha is not equal to checkpoint value!"
    assert (
        checkpoint["loss"] == args.loss
    ), "Error: loss is not equal to checkpoint value!"

    # Loss configuration
    if args.loss == "cross_entropy":
        criterion = regularization.RegularizedLoss(
            net, nn.CrossEntropyLoss(), args.alpha
        )
        one_hot = False
    else:
        criterion = regularization.RegularizedLoss(net, nn.MSELoss(), args.alpha)
        one_hot = True
    print(
        f"==> Current criterion: {criterion.__class__.__name__} with {args.loss} and alpha={args.alpha}"
    )

    # Data
    print("==> Preparing data..")
    batch_size = 512
    num_workers = 2

    if args.data == "CIFAR10":
        assert args.network != "FCN", "CIFAR10 is not supported for FCN"
        data_loader = cifar10.CIFAR10DataLoader(batch_size, num_workers, one_hot)
    else:
        data_loader = mnist.MNISTDataLoader(batch_size, num_workers, one_hot, flatten)
    _, _, test_loader = data_loader.get_data_loaders()

    print("==> Evaluate model loss and hessian..")
    # One batch of train data
    data, target = next(iter(test_loader))
    loss = criterion(net(data.to(device)), target.to(device))

    start = time.time()
    print("-----------Lanzcos algorithm test------------\n")
    if device == "cuda":
        eigvals_lanczos, _ = lanczos.lanczos(
            loss,
            net,
            num_eigenthings=args.num_eigval,
            tol=0,
            use_gpu=True,
        )
    else:
        eigvals_lanczos, _ = lanczos.lanczos(
            loss,
            net,
            num_eigenthings=args.num_eigval,
            tol=0,
        )
    print(
        f"Negative eigval rate by Lanczos algorithms, largest eigvals: {np.sum(eigvals_lanczos < -1e-8)}/ {args.num_eigval} ({np.sum(eigvals_lanczos < -1e-8)/args.num_eigval*100:.3f}%)"
    )
    print(f"Computation time: {time.time() - start:.2f}s\n")


if __name__ == "__main__":
    main()

#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append("../")

from dataloader import CIFAR10  # noqa
from models import VGG16, ResNet50, ShuffleNetV2  # noqa
from src import hessians  # noqa


def main():
    batch_size = 512
    num_workers = 12

    data_loader = CIFAR10.CIFAR10DataLoader(batch_size, num_workers)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()


if __name__ == "__main__":
    main()

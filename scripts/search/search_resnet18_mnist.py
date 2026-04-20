#!/usr/bin/env python3
"""Search GIF unlearning hyperparameters for a trained ResNet18 MNIST model."""

from __future__ import annotations

import sys

from search_mnist_model import main as generic_main


def main() -> None:
    sys.argv = [sys.argv[0], *sys.argv[1:], "--model", "resnet18"]
    generic_main()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    save_path = PROJECT_ROOT / "checkpoints" / "hf_svhn_vgg16.pth"
    command = [
        "python3",
        "scripts/train/train_hf_dataset.py",
        "--dataset",
        "svhn",
        "--model",
        "vgg16",
        "--save-path",
        str(save_path),
    ]
    command.extend(sys.argv[1:])
    raise SystemExit(subprocess.call(command, cwd=PROJECT_ROOT))


if __name__ == "__main__":
    main()

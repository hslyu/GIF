from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    net_name: str
    data: str
    criterion: str
    path: str
    exclusive_label: Optional[int] = None
    lr: float = 0.1
    num_epoch: int = 200
    alpha: float = 0.0
    resume: bool = False
    early_stop: int = 50
    batch_size: int = 512


def fig3_configs(
    net_name: str = "FullyConnectedNet",
    data: str = "MNIST",
    criterion: str = "cross_entropy",
    path: str = "fig3",
    exclusive_label: Optional[int] = None,
    lr: float = 0.1,
    num_epoch: int = 200,
    alpha: float = 0.0,
    resume: bool = False,
    early_stop: int = 50,
    batch_size: int = 512,
):
    return Config(
        net_name,
        data,
        criterion,
        path,
        exclusive_label,
        lr,
        num_epoch,
        alpha,
        resume,
        early_stop,
        batch_size,
    )


def tab1_configs(
    net_name: str = "ResNet18",
    data: str = "MNIST",
    criterion: str = "cross_entropy",
    path: str = "tab1",
    exclusive_label: Optional[int] = None,
    lr: float = 0.1,
    num_epoch: int = 200,
    alpha: float = 0.0,
    resume: bool = False,
    early_stop: int = 50,
    batch_size: int = 512,
    retrained: bool = False,
):
    net_name = net_name + "_retrained" if retrained else net_name
    return Config(
        net_name,
        data,
        criterion,
        path,
        exclusive_label,
        lr,
        num_epoch,
        alpha,
        resume,
        early_stop,
        batch_size,
    )


def tab2_configs(
    net_name: str = "VGG11",
    data: str = "CIFAR10",
    criterion: str = "cross_entropy",
    path: str = "tab2",
    exclusive_label: Optional[int] = None,
    lr: float = 0.1,
    num_epoch: int = 200,
    alpha: float = 0.0,
    resume: bool = False,
    early_stop: int = 50,
    batch_size: int = 512,
    retrained: bool = False,
):
    net_name = net_name + "_retrained" if retrained else net_name
    return Config(
        net_name,
        data,
        criterion,
        path,
        exclusive_label,
        lr,
        num_epoch,
        alpha,
        resume,
        early_stop,
        batch_size,
    )


def tab3_configs():
    return

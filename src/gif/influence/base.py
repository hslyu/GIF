from abc import ABCMeta, abstractmethod

import torch


class InfluenceMethod(metaclass=ABCMeta):
    @abstractmethod
    def compute(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

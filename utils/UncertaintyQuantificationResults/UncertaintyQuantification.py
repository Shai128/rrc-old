import abc
from abc import abstractmethod

import torch

from utils.DataScaler import DataScaler


class UncertaintyQuantification(abc.ABC):
    def __init__(self):
        pass


class UncertaintyQuantificationSet(abc.ABC):
    def __init__(self, x_train, y_train, test_size: int, scaler: DataScaler):
        self.x_train = x_train
        self.y_train = y_train
        self.test_size = test_size
        self.scaler = scaler

    @abstractmethod
    def add_prediction_intervals(self, new_prediction_intervals: UncertaintyQuantification, idx: list, x: torch.Tensor):
        pass
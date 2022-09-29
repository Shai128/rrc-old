import numpy as np
import torch

from utils.DataScaler import DataScaler
from utils.I2IRegressionUQModel import I2IRegressionUQHeuristics
from utils.UncertaintyQuantificationResults.UncertaintyQuantification import UncertaintyQuantification, \
    UncertaintyQuantificationSet


class I2IRegressionUQ(UncertaintyQuantification):
    def __init__(self, intervals: torch.Tensor):
        super().__init__()
        self.intervals = intervals


class I2IRegressionUQSet(UncertaintyQuantificationSet):

    def __init__(self, test_size, scaler: DataScaler, device, width, height, is_calibrated_preds):
        super().__init__(None, None, test_size, scaler)
        self.device = device
        self.intervals = torch.zeros(test_size, width, height, 2)
        if not is_calibrated_preds:
            self.estimated_means = torch.zeros(test_size, width, height)
        else:
            self.estimated_means = None

    def add_prediction_intervals(self, new_prediction_intervals: I2IRegressionUQ, idx, x):
        self.intervals[idx] = new_prediction_intervals.intervals.cpu()

    def add_estimated_mean(self, estimated_mean: torch.Tensor, idx, x):
        self.estimated_means[idx] = estimated_mean.squeeze().cpu().detach()

    @property
    def unscaled_intervals(self):
        return self.intervals

    @property
    def unscaled_mean(self):
        return self.estimated_means


def construct_interval_from_mean_and_heuristics(estimated_mean, uq_heuristics: I2IRegressionUQHeuristics, lambda_hat):
    if torch.is_tensor(uq_heuristics.l):
        l = uq_heuristics.l.squeeze().to(estimated_mean.device)
    elif type(uq_heuristics.l) == np.ndarray:
        l = torch.from_numpy(uq_heuristics.l).squeeze().to(estimated_mean.device)
    else:
        l = uq_heuristics.l
    if torch.is_tensor(uq_heuristics.u):
        u = uq_heuristics.u.squeeze().to(estimated_mean.device)
    elif type(uq_heuristics.u) == np.ndarray:
        u = torch.from_numpy(uq_heuristics.u).squeeze().to(estimated_mean.device)
    else:
        u = uq_heuristics.u

    shape = estimated_mean.squeeze().shape + (2,)
    calibrated_interval = torch.zeros(*shape)
    if isinstance(lambda_hat, np.ndarray):
        lambda_hat = torch.Tensor(lambda_hat)
    if torch.is_tensor(lambda_hat):
        lambda_hat = lambda_hat.to(estimated_mean.device)

    calibrated_interval[..., 0] = estimated_mean.clone() - lambda_hat * l
    calibrated_interval[..., 1] = estimated_mean.clone() + lambda_hat * u
    calibrated_interval = I2IRegressionUQ(calibrated_interval)
    return calibrated_interval



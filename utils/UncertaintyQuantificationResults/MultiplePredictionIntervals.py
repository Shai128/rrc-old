import torch

from utils.DataScaler import DataScaler
from utils.UncertaintyQuantificationResults.UncertaintyQuantification import UncertaintyQuantification, \
    UncertaintyQuantificationSet


class MultiplePredictionIntervals(UncertaintyQuantification):
    def __init__(self, intervals: torch.Tensor):
        super().__init__()
        assert len(intervals.shape) == 3
        self.intervals = intervals


class MultiplePredictionIntervalsSet(UncertaintyQuantificationSet):

    def __init__(self, x_train: torch.Tensor, y_train, test_size, scaler: DataScaler):
        super().__init__(x_train, y_train, test_size, scaler)
        self.device = x_train.device
        y_dim = y_train.shape[-1]
        self.intervals = torch.zeros(test_size, y_dim,  2).to(self.device)

    def add_prediction_intervals(self, new_prediction_intervals: MultiplePredictionIntervals, idx, x):
        self.intervals[idx] = new_prediction_intervals.intervals

    @property
    def unscaled_intervals(self):
        y_lower = self.scaler.unscale_y(self.intervals[:, :, 0])
        y_upper = self.scaler.unscale_y(self.intervals[:, :, 1])
        return torch.cat([y_lower.unsqueeze(-1), y_upper.unsqueeze(-1)], dim=-1)
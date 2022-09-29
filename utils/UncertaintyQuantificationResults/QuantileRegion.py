import torch
import helper
from utils.DataScaler import DataScaler
from utils.UncertaintyQuantificationResults.UncertaintyQuantification import UncertaintyQuantification, \
    UncertaintyQuantificationSet

z_dim_to_quantile_region_sample_size = {
    1: 1e3,
    2: 5e3,
    3: 5e3,
    4: 1e4,
}

class QuantileRegions(UncertaintyQuantification):

    def __init__(self, quantile_region_sample, unscaled_y_train, scaler: DataScaler):
        super().__init__()
        self.quantile_region_sample = quantile_region_sample
        self.in_region_distances = helper.get_min_distance(quantile_region_sample, quantile_region_sample,
                                                           ignore_zero_distance=True,
                                                           y_batch_size=10000,
                                                           points_batch_size=10000)
        self.unscaled_y_train = unscaled_y_train
        self.initial_in_region_threshold = torch.quantile(self.in_region_distances, q=0.8).item()
        self.scaler = scaler
        self.area = self.calc_area()

    def is_in_region(self, points: torch.Tensor, in_region_threshold: float = None):
        if in_region_threshold is None:
            in_region_threshold = self.initial_in_region_threshold
        if len(points.shape) == 2:
            points = points.unsqueeze(1)
            squeeze_dim_1 = True
        else:
            squeeze_dim_1 = False
        res = helper.get_min_distance(points, self.quantile_region_sample,
                                      ignore_zero_distance=False,
                                      y_batch_size=50,
                                      points_batch_size=10000) < in_region_threshold
        if squeeze_dim_1:
            res = res.squeeze(1)
        return res

    def calc_area(self):
        border_max = torch.quantile(self.unscaled_y_train, dim=0, q=0.95) * (11 / 10)
        border_min = torch.quantile(self.unscaled_y_train, dim=0, q=0.05) * (9 / 10)
        y_dim = self.unscaled_y_train.shape[-1]
        n_points_to_sample = z_dim_to_quantile_region_sample_size[y_dim]
        stride = (border_max - border_min) / (n_points_to_sample ** (1 / y_dim))
        device = self.quantile_region_sample.device
        unscaled_y_grid = helper.get_grid_from_borders(border_max, border_min, stride, device)
        y_grid_area = (border_max - border_min).prod(dim=-1)
        scaled_y_grid = self.scaler.scale_y(unscaled_y_grid)
        is_in_region = self.is_in_region(scaled_y_grid.unsqueeze(0).repeat(self.quantile_region_sample.shape[0], 1, 1))
        area = is_in_region.float().mean().item() * y_grid_area
        return area

    def get_area(self):
        return self.area


class QuantileRegionSet(UncertaintyQuantificationSet):
    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor, test_size: int, scaler: DataScaler):
        super().__init__(x_train, y_train, test_size, scaler)
        self.quantile_region_samples = []
        self.quantile_regions = []
        self._areas = []
        self.initial_in_region_threshold = []

    @property
    def areas(self) -> torch.Tensor:
        return torch.stack(self._areas)

    def add_prediction_intervals(self, new_prediction_intervals: QuantileRegions, idx: list, x: torch.Tensor):
        self.quantile_region_samples += [new_prediction_intervals.quantile_region_sample]
        self.quantile_regions += [new_prediction_intervals]
        self._areas += [new_prediction_intervals.area]
        self.initial_in_region_threshold += [new_prediction_intervals.initial_in_region_threshold]

    def is_in_region(self, y_test, in_region_threshold=None, is_scaled=True) -> torch.Tensor:
        if not is_scaled:
            y_test = self.scaler.scale_y(y_test)
        if in_region_threshold is None:
            assert len(self.initial_in_region_threshold) == y_test.shape[0]
            in_region_threshold = torch.Tensor(self.initial_in_region_threshold).to(y_test.device)
        quantile_region_samples = torch.stack([qr.quantile_region_sample for qr in self.quantile_regions]).flatten(0, 1)

        # noinspection PyTypeChecker
        return helper.get_min_distance(y_test.unsqueeze(1), quantile_region_samples,
                                       ignore_zero_distance=False,
                                       y_batch_size=50,
                                       points_batch_size=10000).squeeze() < in_region_threshold
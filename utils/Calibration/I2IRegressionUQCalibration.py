import abc
import copy
from typing import List

import numpy as np

from utils.Calibration.ACICalibration import ACICalibration
from utils.Calibration.StretchingFunctions import StretchingFunction, IdentityStretching
from utils.I2IRegressionUQModel import I2IRegressionUQHeuristics
from utils.TSModel import TSModel
import torch

from utils.UncertaintyQuantificationResults.I2IRegressionUQ import I2IRegressionUQ, \
    construct_interval_from_mean_and_heuristics


class I2ILoss(abc.ABC):
    def __init__(self, nominal_level: float):
        self.nominal_level = nominal_level

    @abc.abstractmethod
    def __call__(self, calibrated_interval_t: I2IRegressionUQ, data) -> (float, dict):
        pass

    @abc.abstractmethod
    def name(self) -> str:
        pass


class ImageMiscoverageLoss(I2ILoss):
    def __call__(self, calibrated_interval_t: I2IRegressionUQ, data) -> (float, dict):
        y_t = data['y'].squeeze()
        valid_idx_mask = data['valid_image_mask'].to(y_t.device).squeeze()
        calibrated_interval_t = calibrated_interval_t.intervals.squeeze().to(y_t.device)[valid_idx_mask]
        y_t = y_t[valid_idx_mask]
        with torch.no_grad():
            err_t_i = 1 - ((y_t >= calibrated_interval_t[..., 0]) &
                           (y_t <= calibrated_interval_t[..., 1])).float()
        loss = err_t_i.mean().item()
        args = {'err_t_i': err_t_i}
        return loss, args

    def name(self) -> str:
        return 'cov_loss'


class ImageCenterMiscoverageLoss(I2ILoss):
    def __init__(self, nominal_level):
        super(ImageCenterMiscoverageLoss, self).__init__(nominal_level)

    def __call__(self, calibrated_interval_t: I2IRegressionUQ, data) -> (float, dict):
        y_t = data['y'].squeeze()
        valid_idx_mask = data['valid_image_mask'].to(y_t.device).squeeze()
        relevant_idx = torch.zeros_like(valid_idx_mask)
        center_pixel = data['center_pixel']
        relevant_idx[center_pixel[0] - 25: center_pixel[0] + 25, center_pixel[1] - 25: center_pixel[1] + 25] = True
        relevant_idx = relevant_idx & valid_idx_mask
        calibrated_interval_t = calibrated_interval_t.intervals.squeeze().to(y_t.device)[relevant_idx]
        y_t = y_t[relevant_idx]
        with torch.no_grad():
            err_t_i = 1 - ((y_t >= calibrated_interval_t[..., 0]) &
                           (y_t <= calibrated_interval_t[..., 1])).float()
        loss = err_t_i.mean().item()
        args = {'err_t_i': err_t_i}
        return loss, args

    def name(self) -> str:
        return 'center_cov_loss'



class ImageCenterLowCoverageLoss(I2ILoss):  # center failure loss
    def __init__(self, nominal_level):
        super(ImageCenterLowCoverageLoss, self).__init__(nominal_level)

    def __call__(self, calibrated_interval_t: I2IRegressionUQ, data) -> (float, dict):
        y_t = data['y'].squeeze()
        valid_idx_mask = data['valid_image_mask'].to(y_t.device).squeeze()
        relevant_idx = torch.zeros_like(valid_idx_mask)
        center_pixel = data['center_pixel']
        relevant_idx[center_pixel[0] - 25: center_pixel[0] + 25, center_pixel[1] - 25: center_pixel[1] + 25] = True
        relevant_idx = relevant_idx & valid_idx_mask
        calibrated_interval_t = calibrated_interval_t.intervals.squeeze().to(y_t.device)[relevant_idx]
        y_t = y_t[relevant_idx]
        with torch.no_grad():
            cov = ((y_t >= calibrated_interval_t[..., 0]) & (y_t <= calibrated_interval_t[..., 1])).float().mean().item()
        if cov <= 0.6:
            err_t = 1
        else:
            err_t = 0
        args = {'image_err_t': err_t}
        return err_t, args

    def name(self) -> str:
        return 'center_cov_loss'


class I2IRegressionUQCalibration(ACICalibration):

    def __init__(self, desired_coverage_level, gamma, calibration_size=1500, model: TSModel = None,
                 stretching_function: StretchingFunction = IdentityStretching(),
                 base_name='',
                 calibration_starting_update_index=0,
                 args=None,
                 **kwargs):
        super().__init__(desired_coverage_level, gamma, calibration_size, calibration_starting_update_index, args)
        self.lambda_t = 1
        self.desired_coverage_level = desired_coverage_level
        self.alpha = 1 - desired_coverage_level
        self.gamma = gamma
        self.calibration_size = calibration_size
        self.model = model
        self.base_name = base_name
        self.phi = stretching_function

    def calibrate(self, data, estimated_mean, regression_uq_heuristics: I2IRegressionUQHeuristics,
                  **kwargs) -> I2IRegressionUQ:
        return construct_interval_from_mean_and_heuristics(estimated_mean, regression_uq_heuristics, self.lambda_t)

    @abc.abstractmethod
    def update(self, data, estimated_mean, regression_uq_heuristics: I2IRegressionUQHeuristics,
               calibrated_interval_t: I2IRegressionUQ,
               **kwargs):
        pass

    def name(self):
        return f"{self.base_name}_{self.phi.name()}_stretching"

    def fit(self, **kwargs):
        pass


class I2IRegressionUQCalibrationSingleTheta(I2IRegressionUQCalibration):

    def __init__(self, desired_coverage_level, gamma, calibration_size=1500, model: TSModel = None,
                 stretching_function: StretchingFunction = IdentityStretching(),
                 base_name='single_theta',
                 **kwargs):
        super().__init__(desired_coverage_level=desired_coverage_level, gamma=gamma, calibration_size=calibration_size,
                         model=model,
                         stretching_function=stretching_function,
                         base_name=base_name,
                         **kwargs)
        self.theta_t = torch.Tensor([0]).to(model.device)
        self.thetas = []
        self.lambdas = []

    def update(self, data, estimated_mean, regression_uq_heuristics: I2IRegressionUQHeuristics,
               calibrated_interval_t: I2IRegressionUQ,
               **kwargs):
        y_t = data['y'].squeeze()
        valid_idx_mask = data['valid_image_mask'].to(y_t.device).squeeze()
        calibrated_interval_t = calibrated_interval_t.intervals.squeeze().to(y_t.device)[valid_idx_mask]
        y_t = y_t[valid_idx_mask]
        with torch.no_grad():
            err_t_i = 1 - ((y_t >= calibrated_interval_t[..., 0]) &
                           (y_t <= calibrated_interval_t[..., 1])).float()
            image_err_t = err_t_i.mean().item()
            self.theta_t = self.theta_t + self.gamma * (image_err_t - self.alpha)
        self.thetas += [self.theta_t.item()]
        args = {'err_t_i': err_t_i, 'image_err_t': image_err_t}
        self.phi.update(self.theta_t, **args)
        self.lambda_t = self.phi(self.theta_t, **args)

        self.lambdas += [self.lambda_t.item()]

    def plot_parameter_vs_time(self, starting_time, ending_time, save_dir=None):
        pass
        # self.plot_parameter_vs_time_aux(self.thetas, starting_time, ending_time, "Theta", save_dir=save_dir)
        # self.plot_parameter_vs_time_aux(self.lambdas, starting_time, ending_time, "Lambdas", save_dir=save_dir)


class I2IRegressionMultiRiskControl(I2IRegressionUQCalibration):

    def __init__(self, desired_coverage_level, gamma, calibration_size=1500, model: TSModel = None,
                 stretching_function: StretchingFunction = IdentityStretching(),
                 base_name='multi_risk',
                 aggregation=None,
                 aggregation_name=None,
                 losses: List[I2ILoss] = [],
                 bound_m=None, bound_M=None,
                 **kwargs):
        if aggregation is None or aggregation_name is None:
            aggregation = lambda l: np.mean(l)
            aggregation_name = 'mean'
        bounded_addition = 'bounded_' if bound_m is not None and bound_M is not None else ''
        base_name = f"{bounded_addition}{base_name}_agg={aggregation_name}"
        super().__init__(desired_coverage_level=desired_coverage_level, gamma=gamma, calibration_size=calibration_size,
                         model=model,
                         stretching_function=stretching_function,
                         base_name=base_name,
                         **kwargs)
        if len(losses) == 0:
            raise Exception("can't handle 0 losses...")
        self.theta_t = [torch.Tensor([0]).to(model.device) for _ in losses]
        self.phis = [copy.deepcopy(self.phi) for _ in losses]
        self.losses = losses
        self.aggregation = aggregation
        self.bound_m = bound_m
        self.bound_M = bound_M

    def update(self, data, estimated_mean, regression_uq_heuristics: I2IRegressionUQHeuristics,
               calibrated_interval_t: I2IRegressionUQ,
               **kwargs):

        lambdas = []
        for i, loss in enumerate(self.losses):
            l_t, args = loss(calibrated_interval_t, data)
            self.theta_t[i] = self.theta_t[i] + self.gamma[i] * (l_t - loss.nominal_level)
            self.phis[i].update(self.theta_t[i], **args)
            lambdas += [self.phis[i](self.theta_t[i], **args).item()]
        self.lambda_t = self.aggregation(lambdas)
        if self.bound_m is not None and self.bound_M is not None:
            upper_exceed = any(theta.item() > self.bound_M for theta in self.theta_t)
            lower_exceed = any(theta.item() < self.bound_m for theta in self.theta_t)
            if upper_exceed:
                self.lambda_t = 99999
            if lower_exceed and not upper_exceed:
                self.lambda_t = 0


    def plot_parameter_vs_time(self, starting_time, ending_time, save_dir=None):
        pass
        # for i in range(len(self.theta_t))
        # self.plot_parameter_vs_time_aux(self.theta_t, starting_time, ending_time, "Theta", save_dir=save_dir)
        # self.plot_parameter_vs_time_aux(self.lambdas, starting_time, ending_time, "Lambdas", save_dir=save_dir)

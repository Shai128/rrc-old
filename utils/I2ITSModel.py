import abc
import copy
import os
import cv2
from torch import nn

from RCOL_datasets.datasets import I2IDataset
from utils.Calibration.I2IRegressionUQCalibration import I2IRegressionUQCalibration
from utils.DepthModel.Leres.lib.models.multi_depth_model_auxiv2 import ModelOptimizer
from utils.I2IRegressionUQModel import OnlineI2IRegressionUQModel
from utils.TSModel import TSModel
from copy import deepcopy
import tqdm
import torch
from typing import List
from utils.Calibration.TSCalibration import TSCalibration
from utils.UncertaintyQuantificationResults.I2IRegressionUQ import I2IRegressionUQSet, \
    construct_interval_from_mean_and_heuristics
import numpy as np

from utils.utils import load_ckpt, save_ckpt, has_state


def unsqueeze_tensors_in_dict(data: dict, device):
    new_data = {}
    for key in data.keys():
        if torch.is_tensor(data[key]):
            new_data[key] = data[key].unsqueeze(0).to(device)
        else:
            new_data[key] = data[key]
    return new_data


# Image to Image Time-Series Model
class I2ITSModel(TSModel, abc.ABC):

    def __init__(self, device, tau, uq_model: OnlineI2IRegressionUQModel, args, dataset: I2IDataset = None):
        super().__init__(dataset, args)
        self.device = device
        self.tau = tau
        self.uq_model = uq_model

    @staticmethod
    @abc.abstractmethod
    def model_name() -> str:
        pass

    def __construct_depth_intervals_online(self, test_idx, coverage_level,
                                           calibrations: List[I2IRegressionUQCalibration] = None,
                                           store_intervals=True,
                                           idx_to_start_storing=0):
        if calibrations is None:
            calibrations = []
        calibration_sizes = [calibration.calibration_size for calibration in calibrations]
        assert len(calibration_sizes) == 0 or min(calibration_sizes) == max(calibration_sizes)
        if self.args.cal_split:
            raise NotImplementedError()

        n_test_points = len(test_idx)
        idx_to_store = np.array(test_idx)[np.array(test_idx) >= idx_to_start_storing]
        idx_not_to_store = np.array(test_idx)[np.array(test_idx) < idx_to_start_storing]

        channels, width, height = self.dataset.x_shape
        if store_intervals:
            uncalibrated_preds = I2IRegressionUQSet(len(idx_to_store), self.scaler, self.device, width, height,
                                                    is_calibrated_preds=False)
            calibration_results = {}
            for calibration in calibrations:
                calibration_results[calibration] = I2IRegressionUQSet(len(idx_to_store), self.scaler, self.device,
                                                                      width, height, is_calibrated_preds=True)
        else:
            uncalibrated_preds, calibration_results = None, None

        idx_to_train = test_idx
        i = 0
        for curr_idx in tqdm.tqdm(idx_to_train):
            data = unsqueeze_tensors_in_dict(self.dataset.get_data(curr_idx, False), self.device)
            with torch.no_grad():
                self.eval()
                self.uq_model.eval()
                estimated_mean, _ = self.estimate_mean(data)
                uq_heuristics = self.uq_model.get_uq_heuristics(data, estimated_mean)
                predicted_interval = construct_interval_from_mean_and_heuristics(estimated_mean,
                                                                                 uq_heuristics, 1)

                if store_intervals and curr_idx >= idx_to_start_storing:
                    idx = list(range(i-len(idx_not_to_store), min(i + 1 - len(idx_not_to_store), n_test_points)))
                    uncalibrated_preds.add_prediction_intervals(predicted_interval, idx, data['rgb'])
                    uncalibrated_preds.add_estimated_mean(estimated_mean, idx, data['rgb'])

                if curr_idx >= self.calibration_starting_update_index:
                    for calibration in calibrations:
                        calibrated_preds = calibration.calibrate(data, estimated_mean, uq_heuristics)
                        if store_intervals and curr_idx >= idx_to_start_storing:
                            calibration_results[calibration].add_prediction_intervals(calibrated_preds, idx,
                                                                                      data['rgb'])
                        calibration.update(data,
                                           estimated_mean=estimated_mean,
                                           regression_uq_heuristics=uq_heuristics,
                                           calibrated_interval_t=calibrated_preds)

            self.train()
            self.uq_model.train()
            data = unsqueeze_tensors_in_dict(self.dataset.get_data(curr_idx, True), self.device)
            data['rgb'].requires_grad = True
            estimated_mean, logit = self.estimate_mean(data)
            self.loss(None, None, None, None, None, None, take_step=True, args=None, data=data,
                      desired_coverage_level=coverage_level, estimated_mean=estimated_mean.detach(), logit=logit)
            i += 1

        return uncalibrated_preds, calibration_results

    def loss(self, x, y, all_pre_x, all_pre_y, q_list, batch_q, take_step, args, data=None, desired_coverage_level=None,
             estimated_mean=None,
             logit=None,
             **kwargs):
        assert data is not None and desired_coverage_level is not None
        uq_loss = self.uq_model.update(data, estimated_mean=estimated_mean.detach())
        return uq_loss + self.loss_aux(all_pre_x, all_pre_y, desired_coverage_level, data=data, take_step=take_step, logit=logit)

    @abc.abstractmethod
    def loss_aux(self, all_pre_x, all_pre_y, desired_coverage_level, **kwargs):
        pass

    def predict_test_online(self, x_train, y_train, x_test, y_test, coverage_level, backward_size,
                            args, save_new_model=False, calibrations: List[TSCalibration] = None, fit_on_train_set=True,
                            train_idx=None, test_idx=None, store_intervals=True, is_train=False, idx_to_start_storing=0):
        models_copy = deepcopy(self.models)
        optimizers_copy = deepcopy(self.optimizers)

        if is_train and "kitti" in args.dataset_name.lower():
            self.train_offline(self.dataset, test_idx[:6000], coverage_level, val_ratio=0., epochs=60)
            test_idx = test_idx[6000:]
        uncalibrated_intervals, calibrated_intervals = self.__construct_depth_intervals_online(test_idx,
                                                                                               coverage_level,
                                                                                               calibrations=calibrations,
                                                                                               store_intervals=store_intervals,
                                                                                               idx_to_start_storing=idx_to_start_storing)

        if not save_new_model:
            self.update_models(models_copy)
            for i in range(len(self.models)):
                self.models[i] = models_copy[i]
            for i in range(len(self.optimizers)):
                self.optimizers[i] = optimizers_copy[i]

        return uncalibrated_intervals, calibrated_intervals

    def train_offline(self, dataset, train_idx, coverage_level, val_ratio=0.2, epochs=60):
        dataset_name = self.args.dataset_name
        train_size = int(len(train_idx) * (1 - val_ratio))
        epochs_to_train = epochs
        for e in range(epochs, 0, -20):
            if self.uq_model.has_state(dataset_name, step=e * train_size, epoch=e) and\
                    self.has_state(dataset_name, step=e * train_size, epoch=e):
                self.load_state(dataset_name, step=e * train_size, epoch=e)
                self.uq_model.load_state(dataset_name, step=e * train_size, epoch=e)
                epochs_to_train = epochs - e
                break

        original_train_idx = copy.deepcopy(train_idx)
        np.random.shuffle(original_train_idx)
        val_idx = train_idx[:int(val_ratio * len(train_idx))]
        train_idx = train_idx[int(val_ratio * len(train_idx)):]

        wait = 10
        epochs_since_last_best_loss = 0
        best_loss = -np.inf

        for e in (range(epochs_to_train)):
            self.train()
            self.uq_model.train()
            np.random.shuffle(train_idx)
            for i in tqdm.tqdm(train_idx):
                data = unsqueeze_tensors_in_dict(dataset.get_data(i, True), self.device)
                data['rgb'].requires_grad = True
                estimated_mean, logit = self.estimate_mean(data)
                self.loss(None, None, None, None, None, None, take_step=True, args=None, data=data,
                          desired_coverage_level=coverage_level, estimated_mean=estimated_mean,
                          logit=logit)

            if len(val_idx) > 0:
                self.eval()
                self.uq_model.eval()
                val_loss = self.validate(dataset, val_idx, coverage_level)
                if val_loss < best_loss:
                    epochs_since_last_best_loss += 1
                    best_loss = val_loss
                    if epochs_since_last_best_loss >= wait:
                        break
                else:
                    epochs_since_last_best_loss = 0
            if e % 20 == 0 and e > 0:
                self.save_state(dataset_name, e * len(train_idx), e)
                self.uq_model.save_state(dataset_name, e * len(train_idx), e)
            print("epoch: ", e)

        self.save_state(dataset_name, epochs * len(train_idx), epochs)
        self.uq_model.save_state(dataset_name, epochs * len(train_idx), epochs)

    def validate(self, dataset, val_idx, coverage_level) -> float:

        losses = []
        for i in tqdm.tqdm(val_idx):
            data = unsqueeze_tensors_in_dict(dataset.get_data(i, False), self.device)
            estimated_mean = self.estimate_mean(data)
            losses += [self.loss(None, None, None, None, None, None, take_step=True, args=None, data=data,
                                 desired_coverage_level=coverage_level,
                                 estimated_mean=estimated_mean.detach()).cpu().item()]

        return np.item(np.mean(losses))

    def load_state(self, dataset_name, step, epoch):
        return load_ckpt(dataset_name, self.model_name(), self.network,
                         optimizer=self.optimizer.optimizer, scheduler=self.scheduler,
                         step=step, epoch=epoch)

    def save_state(self, dataset_name, step, epoch):
        save_ckpt(dataset_name, self.model_name(), step, epoch, self.network,
                  optimizer=self.optimizer.optimizer, scheduler=self.scheduler)

    def has_state(self, dataset_name, step, epoch):
        return has_state(dataset_name, self.model_name(), step, epoch)

    def _TSModel__construct_interval_aux(self, xi, previous_ys, desired_coverage, true_y=None, data=None):
        pass

    def initialize_scalers(self, x_train, y_train):
        pass

    def get_uncertainty_quantification_set_class(self):
        pass

    def plot_losses(self):
        pass

    @property
    @abc.abstractmethod
    def optimizer(self) -> ModelOptimizer:
        pass

    @property
    @abc.abstractmethod
    def scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        pass

    @property
    @abc.abstractmethod
    def network(self) -> nn.Module:
        pass



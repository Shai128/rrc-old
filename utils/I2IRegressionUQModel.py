import abc
from abc import ABC
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from utils.DepthModel.Leres.lib.models.multi_depth_model_auxiv2 import ModelOptimizer, ModelLoss
from utils.DepthModel.Leres.lib.utils.evaluate_depth_error import recover_metric_depth
from utils.DepthModel.Leres.lib.utils.lr_scheduler_custom import make_lr_scheduler
from utils.TSQR import batch_pinball_loss
from utils.utils import save_ckpt, load_ckpt, has_state
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
import torch.nn.functional as F


class I2IRegressionUQHeuristics:
    def __init__(self, l, u):
        self.l = l
        self.u = u


def dummy_loss(device):
    loss = torch.Tensor([0]).to(device)
    loss.requires_grad = True
    return loss


def get_feature_points_mask(valid_image_mask: torch.Tensor, image: np.ndarray, batch_size) -> torch.Tensor:
    valid_idx = valid_image_mask.nonzero()
    n_valid_idx = valid_idx.shape[0]
    feature_points_mask = torch.zeros_like(valid_image_mask, dtype=torch.bool)
    if n_valid_idx == 0:
        return feature_points_mask

    rnd_points = np.random.choice(n_valid_idx, (batch_size,))
    feature_points_mask[valid_idx[rnd_points][:, 0], valid_idx[rnd_points][:, 1]] = 1

    feature_points_mask = feature_points_mask & valid_image_mask

    return feature_points_mask


# def get_good_feature_points_mask(valid_image_mask: torch.Tensor, image: np.ndarray, batch_size) -> torch.Tensor:
#     valid_idx = valid_image_mask.nonzero()
#     n_valid_idx = valid_idx.shape[0]
#     feature_points_mask = torch.zeros_like(valid_image_mask, dtype=torch.bool)
#     if n_valid_idx == 0:
#         return feature_points_mask
#
#     valid_idx_mask_numpy = valid_image_mask.detach().cpu().numpy()
#     mask = torch.ones_like(valid_image_mask) & valid_image_mask
#     mask = mask.detach().cpu().numpy()
#     min_x = valid_idx_mask_numpy.nonzero()[0].min().item()
#     min_y = valid_idx_mask_numpy.nonzero()[1].min().item()
#     max_x = valid_idx_mask_numpy.nonzero()[0].max().item()
#     max_y = valid_idx_mask_numpy.nonzero()[1].max().item()
#     mask[min_x:min_x + 10, :] = 0
#     mask[:, min_y:min_y + 10] = 0
#     mask[max_x - 10:max_x, :] = 0
#     mask[:, max_y - 10:max_y] = 0
#
#     good_feature_points = torch.from_numpy(cv2.goodFeaturesToTrack(image.astype(np.float32), batch_size, 0.0001, 20, mask=mask.astype(np.uint8))).squeeze().long()
#     feature_points_mask[good_feature_points[:,0], good_feature_points[:, 1]] = 1
#
#     feature_points_mask = feature_points_mask & valid_image_mask
#
#     return feature_points_mask

class I2IRegressionUQModel(nn.Module, ABC):
    def __init__(self, desired_coverage_level):
        super().__init__()
        self.desired_coverage_level = desired_coverage_level

    @abc.abstractmethod
    def get_uq_heuristics(self, data, estimated_mean) -> I2IRegressionUQHeuristics:
        pass


class OnlineI2IRegressionUQModel(I2IRegressionUQModel, ABC):
    def __init__(self, desired_coverage_level, loss_batch_size):
        super().__init__(desired_coverage_level)
        self.losses = []
        self.loss_batch_size = loss_batch_size
        self.alpha = 1 - desired_coverage_level

    def update(self, data, estimated_mean):
        loss = self.loss(data, estimated_mean)
        loss_dict = {'total_loss': loss}
        self.optimizer.optim(loss_dict)
        self.losses += [loss.cpu().item()]
        return loss

    def load_state(self, dataset_name, step, epoch):
        return load_ckpt(dataset_name, self.model_save_name, self.network,
                         optimizer=self.optimizer.optimizer, scheduler=self.scheduler,
                         step=step, epoch=epoch)

    def save_state(self, dataset_name, step, epoch):
        save_ckpt(dataset_name, self.model_save_name, step, epoch, self.network,
                  optimizer=self.optimizer.optimizer, scheduler=self.scheduler)

    def has_state(self, dataset_name, step, epoch):
        return has_state(dataset_name, self.model_save_name, step, epoch)

    @abc.abstractmethod
    def loss(self, data, estimated_mean):
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

    def plot_loss(self):
        plt.plot(self.losses)
        plt.xlabel("Time")
        plt.ylabel("Loss")
        plt.show()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    def model_save_name(self) -> str:
        return self.name


class NonLearningUQModel(OnlineI2IRegressionUQModel, abc.ABC):

    def __init__(self, desired_coverage_level, loss_batch_size, **kwargs):
        super().__init__(desired_coverage_level, loss_batch_size)

    def loss(self, data, estimated_mean):
        return dummy_loss(data['y'].device)

    def update(self, data, estimated_mean):
        return self.loss(data, estimated_mean)

    @property
    def optimizer(self):
        return None

    @property
    def scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return None

    @property
    def network(self) -> nn.Module:
        return None

    def load_state(self, dataset_name, step, epoch):
        pass

    def save_state(self, dataset_name, step, epoch):
        pass

    def has_state(self, dataset_name, step, epoch):
        return True


class BaselineUQModel(NonLearningUQModel):

    def __init__(self, desired_coverage_level, loss_batch_size, **kwargs):
        super().__init__(desired_coverage_level, loss_batch_size, **kwargs)

    def get_uq_heuristics(self, data, estimated_mean) -> I2IRegressionUQHeuristics:
        return I2IRegressionUQHeuristics(1, 1)

    @property
    def name(self) -> str:
        return "baseline"


class PreviousResidualUQModel(NonLearningUQModel):

    def __init__(self, desired_coverage_level, loss_batch_size, **kwargs):
        super().__init__(desired_coverage_level, loss_batch_size, **kwargs)
        self.prev_ls = []
        self.prev_us = []

    def get_uq_heuristics(self, data, estimated_mean) -> I2IRegressionUQHeuristics:
        residual = estimated_mean.squeeze() - data['y'].squeeze()
        curr_l = residual * (residual > 0)
        curr_u = - residual * (residual < 0)

        if len(self.prev_ls) == 0:
            l = u = 1
        else:
            l = torch.stack(self.prev_ls).mean(dim=0)
            u = torch.stack(self.prev_us).mean(dim=0)
            mask = data['feature_points_mask'].squeeze()
            if curr_l[mask].sum() > 20 and curr_u[mask].sum() > 20:
                l = recover_metric_depth(l, curr_l, mask0=mask)
                u = recover_metric_depth(u, curr_u, mask0=mask)

        heuristics = I2IRegressionUQHeuristics(l, u)

        self.prev_ls += [curr_l]
        self.prev_us += [curr_u]
        self.prev_ls = self.prev_ls[-10:]
        self.prev_us = self.prev_us[-10:]
        return heuristics
    @property
    def name(self) -> str:
        return "previous_residual"


def register_using_optical_flow(image0, image1):
    device = 'cpu'
    if torch.is_tensor(image0):
        device = image0.device
        image0 = image0.cpu().detach().numpy()
    if torch.is_tensor(image1):
        image1 = image1.cpu().detach().numpy()
    nr, nc = image0.shape
    v, u = optical_flow_tvl1(image1, image0, num_iter=2, num_warp=1)
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                         indexing='ij')
    image0_warp = warp(image0, np.array([row_coords + v, col_coords + u]), mode='edge')

    return torch.from_numpy(image0_warp).to(device)


class PreviousResiduasWithFlowlUQModel(NonLearningUQModel):

    def __init__(self, desired_coverage_level, loss_batch_size, **kwargs):
        super().__init__(desired_coverage_level, loss_batch_size, **kwargs)
        self.prev_depths = []
        self.prev_depth_estimates = []

    def get_uq_heuristics(self, data, estimated_mean) -> I2IRegressionUQHeuristics:
        pad = data['augmentation_info']['pad']
        m, n = estimated_mean.squeeze().shape
        residual = (estimated_mean.squeeze() - data['y'].squeeze())[pad[0]: m - pad[1], pad[2]: n - pad[3]]
        curr_l = residual * (residual > 0)
        curr_u = - residual * (residual < 0)
        if len(self.prev_depths) == 0:
            l = u = 1
        else:
            for i in range(len(self.prev_depths) - 1):
                self.prev_depths[i] = register_using_optical_flow(self.prev_depths[i], self.prev_depths[-1])
                self.prev_depth_estimates[i] = register_using_optical_flow(self.prev_depth_estimates[i],
                                                                           self.prev_depth_estimates[-1])
            prev_residuals = torch.stack(self.prev_depth_estimates) - torch.stack(self.prev_depths)

            l = (prev_residuals * (prev_residuals > 0)).mean(dim=0)
            u = (- prev_residuals * (prev_residuals < 0)).mean(dim=0)

            mask = data['feature_points_mask'].squeeze()[pad[0]: m - pad[1], pad[2]: n - pad[3]]
            if curr_l[mask].sum() > 20 and curr_u[mask].sum() > 20:
                l = torch.from_numpy(recover_metric_depth(l, curr_l, mask0=mask)).to(l.device)
                u = torch.from_numpy(recover_metric_depth(u, curr_u, mask0=mask)).to(u.device)
            l_padded = torch.zeros_like(estimated_mean)
            u_padded = torch.zeros_like(estimated_mean)
            l_padded[pad[0]: m - pad[1], pad[2]: n - pad[3]] = l
            u_padded[pad[0]: m - pad[1], pad[2]: n - pad[3]] = u
            l, u = l_padded, u_padded

        heuristics = I2IRegressionUQHeuristics(l, u)

        self.prev_depths += [data['y'].squeeze()[pad[0]: m - pad[1], pad[2]: n - pad[3]]]
        self.prev_depth_estimates += [estimated_mean.squeeze()[pad[0]: m - pad[1], pad[2]: n - pad[3]]]
        self.prev_depths = self.prev_depths[-5:]
        self.prev_depth_estimates = self.prev_depth_estimates[-5:]
        return heuristics

    @property
    def name(self) -> str:
        return "previous_residual_with_flow"


class ResidualMagnitudeRegression(OnlineI2IRegressionUQModel):

    @property
    def network(self) -> torch.nn.Module:
        return self._network

    def __init__(self, desired_coverage_level, pre_trained_network_without_last_layer, model_out_channels, device,
                 loss_batch_size):
        super().__init__(desired_coverage_level, loss_batch_size)
        self._network = torch.nn.Sequential(pre_trained_network_without_last_layer.to(device),
                                            nn.Conv2d(model_out_channels, 1, kernel_size=3, padding=1, stride=1,
                                                      bias=True).to(device),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True).to(device))
        self.optimizer = ModelOptimizer(self)
        self.optimizers = [self.optimizer]
        self._scheduler = make_lr_scheduler(optimizer=self.optimizer.optimizer)

    def get_uq_heuristics(self, data, estimated_mean) -> I2IRegressionUQHeuristics:
        residual = self.forward(data)
        residual = recover_metric_depth(residual, (data['depth'].squeeze() - estimated_mean.squeeze()).abs(),
                                        mask0=data['feature_points_mask'].squeeze())
        return I2IRegressionUQHeuristics(residual.squeeze(), residual.squeeze())

    def forward(self, data):
        network_out = self.network(data['rgb'])
        return network_out

    def loss(self, data, estimated_mean):
        batch_size = self.loss_batch_size
        feature_points_mask = get_feature_points_mask(data['valid_image_mask'].squeeze(),
                                                      data['y'].squeeze().cpu().detach().numpy(), batch_size)
        if feature_points_mask.float().sum() == 0:
            return dummy_loss(data['y'].device)
        pred = self.forward(data).squeeze()
        estimated_mean = estimated_mean.detach().squeeze()[feature_points_mask]
        ground_truth = data['y'].squeeze()[feature_points_mask]
        return ((pred[feature_points_mask] - (estimated_mean - ground_truth).abs()).abs()).mean()

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def name(self) -> str:
        return "residual_magnitude"

    @network.setter
    def network(self, value):
        self._network = value

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        self._scheduler = value

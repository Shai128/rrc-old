from argparse import Namespace
from torch import nn
from RCOL_datasets.datasets import LazyDepthDataSet
from utils.DepthModel.Leres.lib.models.multi_depth_model_auxiv2 import RelDepthModel, ModelOptimizer
from utils.DepthModel.Leres.lib.utils.evaluate_depth_error import recover_metric_depth
from utils.DepthModel.Leres.lib.utils.lr_scheduler_custom import make_lr_scheduler_from_cfg
from utils.DepthModel.Leres.lib.utils.net_tools import load_model_ckpt
from utils.I2IRegressionUQModel import OnlineI2IRegressionUQModel
from utils.I2ITSModel import I2ITSModel
import torch
from utils.DepthModel.Leres.lib.configs.config import cfg


class DepthQR(I2ITSModel):

    def model_name(self) -> str:
        return f'depth_qr_backbone={self.backbone}'

    def __init__(self, trained_model_path, backbone, uq_model: OnlineI2IRegressionUQModel,
                 device, tau, args, dataset: LazyDepthDataSet = None):
        super().__init__(device=device, tau=tau, dataset=dataset, args=args, uq_model=uq_model)

        self.network = RelDepthModel(device, backbone=backbone)
        self.backbone = backbone
        self.optimizer = ModelOptimizer(self.network)
        self.optimizers = [self.optimizer]
        self.scheduler = make_lr_scheduler_from_cfg(cfg=cfg, optimizer=self.optimizer.optimizer)
        train_args = Namespace(load_ckpt=trained_model_path, resume=True)
        load_model_ckpt(train_args, self.network, None, None)
        self.models = [self.network]
        self.device = device
        self.tau = tau
        self.losses = []

    def update_models(self, models):
        self.models = models
        self.network = self.models[0]
        self.optimizer = ModelOptimizer(self.network)
        self.scheduler = make_lr_scheduler_from_cfg(cfg=cfg, optimizer=self.optimizer.optimizer)
        self.optimizers = [self.optimizer]

    def estimate_mean(self, data):
        out = self.network.inference(data)
        pred_depth = out['pred_depth'].squeeze().detach()
        logit = out['logit']
        pred_depth = recover_metric_depth(pred_depth, data['depth'].squeeze(), mask0=data['feature_points_mask'])
        pred_depth = torch.Tensor(pred_depth).to(data['rgb'].device)
        return pred_depth, logit

    def initialize_scalers(self, x_train, y_train):
        pass

    def get_uncertainty_quantification_set_class(self):
        pass

    def plot_losses(self):
        pass

    def _TSModel__construct_interval_aux(self, **kwargs):
        pass

    def loss_aux(self, all_pre_x, all_pre_y, desired_coverage_level, data=None, take_step=True, logit=None):
        out = self.network(data,logit=logit)
        losses_dict = out['losses']
        self.losses += [losses_dict['total_loss'].cpu().item()]
        if take_step:
            self.optimizer.optim(losses_dict)
        self.scheduler.step()
        return losses_dict['total_loss'].cpu().item()

    @property
    def network(self) -> nn.Module:
        return self._network

    @property
    def optimizer(self) -> ModelOptimizer:
        return self._optimizer

    @property
    def scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return self._scheduler

    @network.setter
    def network(self, value):
        self._network = value

    @optimizer.setter
    def optimizer(self, value: ModelOptimizer):
        self._optimizer = value

    @scheduler.setter
    def scheduler(self, value):
        self._scheduler = value


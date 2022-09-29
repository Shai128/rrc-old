import abc

import torch
import numpy as np


class OutputToPredSetModel(abc.ABC):
    def to_pred_set(self, model_outputs):
        """
        Get the underlined model outputs `model_outputs` and returns prediction sets for each of the outputs.
        """
        raise NotImplementedError()

    def update(self, losses: torch.FloatTensor):
        """
        Gets a tensor of losses, `losses` of shape batch_size x risks_num and updates the model accordingly.
        """
        raise NotImplementedError()


class SingleScalarOutputToPredSetModel(OutputToPredSetModel, abc.ABC):
    def __init__(self, desired_risk: float, *, gamma=0.05):
        self.desired_risk = desired_risk
        self.gamma = gamma
        self.theta = 0

    def update(self, losses: torch.FloatTensor):
        assert len(losses.shape) == 2 and losses.shape[1] == 1
        batch_size = losses.shape[0]
        loss_to_desired_risk_diff = losses.sum(dim=0).item() - self.desired_risk*batch_size
        diff = self.gamma*(loss_to_desired_risk_diff)
        # diff *= np.exp(-1/(np.abs(loss_to_desired_risk_diff)*10))
        self.theta = self.theta + diff


class SingleScalarWithClipOutputToPredSetModel(OutputToPredSetModel, abc.ABC):
# class SingleScalarOutputToPredSetModel(OutputToPredSetModel, abc.ABC):
    def __init__(self, desired_risk: float, *, gamma=0.05):
        self.desired_risk = desired_risk
        self.gamma = gamma
        self.theta = 0
        self.count = 0

    def update(self, losses: torch.FloatTensor):
        assert len(losses.shape) == 2 and losses.shape[1] == 1
        self.count += 1
        if self.count % 20 != 0:
            return
        batch_size = losses.shape[0]
        loss_to_desired_risk_diff = losses.sum(dim=0).item() - self.desired_risk * batch_size
        diff = self.gamma*(loss_to_desired_risk_diff)
        # diff *= np.exp(-1 / (np.abs(loss_to_desired_risk_diff) * 10))
        if diff > 0:
            print('diff_sign=+1')
        elif diff < 0:
            print('diff_sign=-1')
        else:
            print('diff_sign=0')
        self.theta = self.theta + diff#np.clip(diff, -0.0001, 0.0001)
        print(f'theta={self.theta}')


class SingleScalarOutputToPredSetModel(OutputToPredSetModel, abc.ABC):
    def __init__(self, desired_risk: float, *, gamma=0.05, history_size=1, update_every=1):
        self.desired_risk = desired_risk
        self.gamma = gamma
        self.theta = 0
        self.history = []
        self.history_size = history_size
        self.update_every = update_every
        self.count = 0

    def update(self, losses: torch.FloatTensor):
        assert len(losses.shape) == 2 and losses.shape[1] == 1
        self.count += 1
        if self.count % self.update_every != 0:
            return

        batch_size = losses.shape[0]
        loss_to_desired_risk_diff = losses.sum(dim=0).item() - self.desired_risk*batch_size

        self.history.append(loss_to_desired_risk_diff)
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]
        loss_to_desired_risk_diff_mean = np.mean(self.history)
        diff = self.gamma*(loss_to_desired_risk_diff_mean)
        # diff *= np.exp(-1/(np.abs(loss_to_desired_risk_diff)*10))
        self.theta = self.theta + diff


class SingleScalarOutputToPredSetModel(OutputToPredSetModel, abc.ABC):
    def __init__(self, desired_risk: float, *, gamma=0.05, history_size=1, update_every=1):
        self.desired_risk = desired_risk
        self.gamma = gamma
        self.theta = 0
        self.history = []
        self.history_size = history_size
        self.update_every = update_every
        self.count = 0

    def update(self, losses: torch.FloatTensor):
        assert len(losses.shape) == 2 and losses.shape[1] == 1
        self.count += 1
        if self.count % self.update_every != 0:
            return

        batch_size = losses.shape[0]
        loss_to_desired_risk_diff = losses.sum(dim=0).item() - self.desired_risk*batch_size

        self.history.append(loss_to_desired_risk_diff)
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]
        loss_to_desired_risk_diff_mean = np.mean(self.history)
        diff = self.gamma*(loss_to_desired_risk_diff_mean)
        # diff *= np.exp(-1/(np.abs(loss_to_desired_risk_diff)*10))
        self.theta = self.theta + diff


class RiskLoss(abc.ABC):
    def calc(self, y, y_tag, y_history=None, y_tag_history=None) -> torch.FloatTensor:
        """
        Gets samples labels `y` and their prediction sets `y_tag` (as batches) and returns a tensor of losses,
        of shape batch_size x risks_num.

        Note: Don't be confused between the `y` and `model_outputs`. In CQR for example, the output label `y`
        (a single scalar) is not the same as the model output (2 scalars), which is the same shape as the final
        prediction set (but different in values).
        """
        raise NotImplementedError()


class RiskCalibration:
    def __init__(self,
                 tau: OutputToPredSetModel,
                 risks_losses: RiskLoss,
                 ):
        """
        `loss_fn` gets model outputs and their prediction sets (as batches) and returns a tensor of losses,
        of shape batch_size x risks_num
        """
        self.tau = tau
        self.risks_losses = risks_losses

    def to_pred_set(self, model_outputs):
        """
        Get the underlined model outputs `model_outputs` and returns prediction sets for each of the outputs.
        """
        return self.tau.to_pred_set(model_outputs)

    def update(self, y, uncalibrated_pred_sets):
        """
        Updates the underlined parameters
        """
        pred_sets = self.tau.to_pred_set(uncalibrated_pred_sets)
        losses = self.risks_losses.calc(y.unsqueeze(0), pred_sets.unsqueeze(0))
        self.tau.update(losses)

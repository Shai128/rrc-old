import abc
from typing import Type, Dict

import numpy as np
import torch
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, OrdinalHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario

import utils.Calibration.StretchingFunctions
from utils.Calibration.RiskCalibration import RiskLoss
from utils.TSQR import batch_pinball_loss


class OutputToPredSet(abc.ABC):
    def to_pred_set(self, model_outputs, params):
        """
        Get the underlined model outputs `model_outputs` and `params` and returns prediction sets for each of the outputs.
        """
        raise NotImplementedError()


class PhiFunction:
    def __init__(self, **kwargs):
        pass

    def calc(self, theta: float, **kwargs) -> float:
        raise NotImplementedError()

    def update(self, **kwargs):
        pass  # Usually does nothing

    @classmethod
    def get_params_space(cls):
        return []


class IdentityPhiFunction(PhiFunction):
    def calc(self, theta: float, **kwargs) -> float:
        return theta


class ExpPhiFunction(PhiFunction):
    def calc(self, theta: float, **kwargs) -> float:
        if theta > 0:
            return np.exp(theta) - 1
        return -np.exp(-theta) + 1


class PhiRiskCalibration:
    def __init__(self,
                 desired_risk: float,
                 tau: OutputToPredSet,
                 risks_losses: RiskLoss,
                 phi_type: Type[PhiFunction],
                 phi_params: Dict,
                 update_phi_after: int,
                 *, gamma=0.05, risks_loss_history_size=0, search_params=True,
                 ):
        self.desired_risk = desired_risk
        self.tau = tau
        self.risks_losses = risks_losses
        self.phi_type = phi_type
        self.phi_params = phi_params
        self.update_phi_after = update_phi_after
        self.phi = None
        self.theta = 0.0
        self.gamma = gamma
        self.search_params = search_params
        self.count = 0
        self.risks_loss_history_size = risks_loss_history_size
        self.risks_loss_y_history = []
        self.risks_loss_y_tag_history = []
        self.y_history = []
        self.uncalibrated_pred_sets_history = []

    def to_pred_set(self, model_outputs):
        """
        Get the underlined model outputs `model_outputs` and returns prediction sets for each of the outputs.
        """
        if self.count <= self.update_phi_after:
            return model_outputs
        phi_theta = self.phi.calc(self.theta)
        return self.tau.to_pred_set(model_outputs, self._phi_theta_after_safeguard(phi_theta))

    def update(self, y, uncalibrated_pred_sets):
        """
        Updates the underlined parameters
        """
        self.count += 1

        y = y.cpu()
        uncalibrated_pred_sets = uncalibrated_pred_sets.cpu()

        if self.count <= self.update_phi_after:
            self.y_history.append(y)
            self.uncalibrated_pred_sets_history.append(uncalibrated_pred_sets)

        if self.count == self.update_phi_after:
            y_stacked = torch.stack(self.y_history)
            self.change_mean = float(torch.abs(y_stacked[1:] - y_stacked[:-1]).mean())
            if self.search_params:
                params_space = self.phi_type.get_params_space()
                self.find_best_params(y_stacked, torch.stack(self.uncalibrated_pred_sets_history), params_space)
            else:
                self.phi = self.phi_type(**self.phi_params, change_mean=self.change_mean)
            self.y_history = None
            self.uncalibrated_pred_sets_history = None

        if self.count <= self.update_phi_after:
            return

        phi_theta = self.phi.calc(self.theta)
        pred_sets = self.tau.to_pred_set(uncalibrated_pred_sets.unsqueeze(0), self._phi_theta_after_safeguard(phi_theta)).squeeze(0)
        y_history = None
        y_tag_history = None
        if self.risks_loss_history_size > 0:
            self.risks_loss_y_history.append(y.unsqueeze(0))
            self.risks_loss_y_tag_history.append(pred_sets.unsqueeze(0))
            y_history = torch.stack(self.risks_loss_y_history[-self.risks_loss_history_size:], dim=1)
            y_tag_history = torch.stack(self.risks_loss_y_tag_history[-self.risks_loss_history_size:], dim=1)
        losses = self.risks_losses.calc(y.unsqueeze(0), pred_sets.unsqueeze(0), y_history, y_tag_history)

        assert len(losses.shape) == 2 and losses.shape[1] == 1

        batch_size = losses.shape[0]
        loss_to_desired_risk_diff = losses.sum(dim=0).item() - self.desired_risk * batch_size
        self.theta += self.gamma * loss_to_desired_risk_diff

        self.phi.update(
            y=y,
            uncalibrated_pred_sets=uncalibrated_pred_sets,
            calibrated_pred_sets=pred_sets,
            loss_to_desired_risk_diff=loss_to_desired_risk_diff,
            tau=self.tau,
            risks_losses=self.risks_losses,
            desired_risk=self.desired_risk,
            gamma=self.gamma,
        )

    def _phi_theta_after_safeguard(self, phi_theta):
        if self.theta < -10000:
            after_safeguard = -1e10
            # print(f'Theta is too low, setting it to {after_safeguard}')
        elif self.theta > 10000:
            after_safeguard = 1e10
            # print(f'Theta is too high, setting it to {after_safeguard}')
        else:
            after_safeguard = phi_theta
        # print(f'phi_theta={phi_theta:.3f} after_safeguard={after_safeguard:.3f}')
        return after_safeguard

    def find_best_params(self, y, uncalibrated_pred_sets, params_space):
        configspace = ConfigurationSpace()
        for param in params_space:
            configspace.add_hyperparameter(param)
        # configspace.add_hyperparameter(UniformFloatHyperparameter("gamma", lower=0.001, upper=0.1))
        possible_gammas = self.gamma if type(self.gamma) is list else [self.gamma]
        configspace.add_hyperparameter(OrdinalHyperparameter('gamma', sequence=possible_gammas))

        # Provide meta data for the optimization
        scenario = Scenario({
            "run_obj": "quality",  # Optimize quality (alternatively runtime)
            "runcount-limit": len(possible_gammas) if len(params_space) == 0 else 40,  # Max number of function evaluations (the more the better)
            "cs": configspace,
            # "multi_objectives": "pinball_loss",
            "limit_resources": False,
        })

        smac = SMAC4BB(scenario=scenario, tae_runner=self.run_phi_for_config)
        self.ctx = y, uncalibrated_pred_sets, self.tau, self.risks_losses, self.desired_risk
        best_found_config = smac.optimize()
        print(self.phi_type.__name__, 'best_found_config:', best_found_config)
        self.gamma = best_found_config["gamma"]
        self.phi = self.phi_type(**self.phi_params, **best_found_config.get_dictionary(), change_mean=self.change_mean)

    def run_phi_for_config(self, config):
        y, uncalibrated_pred_sets, tau, risks_losses, desired_risk = self.ctx
        gamma = config['gamma']
        phi = self.phi_type(**self.phi_params, **config.get_dictionary(), change_mean=self.change_mean)

        theta = 0.0
        pred_set_list = []
        for i in range(len(y)):
            phi_theta = phi.calc(theta)
            pred_sets = tau.to_pred_set(uncalibrated_pred_sets[i].unsqueeze(0), phi_theta)
            pred_set_list.append(pred_sets.squeeze(0))
            start_idx = max(0, i-40+1)
            losses = risks_losses.calc(y[i].unsqueeze(0), pred_sets, y[start_idx:i+1].unsqueeze(0), torch.stack(pred_set_list[start_idx:i+1], dim=0).unsqueeze(0))
            assert len(losses.shape) == 2 and losses.shape[1] == 1
            batch_size = losses.shape[0]
            loss_to_desired_risk_diff = losses.sum(dim=0).item() - desired_risk * batch_size
            theta += gamma * loss_to_desired_risk_diff
            phi.update(y=y[i], calibrated_pred_sets=pred_sets[0], loss_to_desired_risk_diff=loss_to_desired_risk_diff, gamma=self.gamma)

        pred_sets = torch.stack(pred_set_list)
        if len(y.shape) == 3:
            y = y[:, -1, :]
            pred_sets = pred_sets[:, -1, :]

        lower_pinball_loss = float(batch_pinball_loss(0.05, pred_sets[:, 0], y))
        upper_pinball_loss = float(batch_pinball_loss(0.95, pred_sets[:, 1], y))
        avg_pinball_loss = (lower_pinball_loss + upper_pinball_loss) / 2
        return avg_pinball_loss

from typing import Optional

import numpy as np
from ConfigSpace import UniformFloatHyperparameter

import utils.Calibration.StretchingFunctions
from utils.Calibration.PhiRiskCalibration import PhiFunction, ExpPhiFunction, IdentityPhiFunction
from utils.Calibration.ScoreFunction import ScoreFunction


class LossAwarePhiFunction(PhiFunction):
    def __init__(self, change_mean, beta1, beta2, score_func: Optional[ScoreFunction] = None, with_exp_phi=False, **kwargs):
        super().__init__(**kwargs)
        self.low = -change_mean
        self.high = change_mean
        self.beta1 = beta1
        self.beta2 = beta2
        self.score_func = score_func
        self.theta_tag = 0.0
        self.underline_phi = ExpPhiFunction() if with_exp_phi else IdentityPhiFunction()

    def calc(self, theta: float, **kwargs) -> float:
        return self.underline_phi.calc(theta) + self.theta_tag

    def update(
            self,
            y,
            calibrated_pred_sets,
            loss_to_desired_risk_diff: float,
            **kwargs,
    ):
        d = self.beta1 * np.exp(self.beta2 * abs(loss_to_desired_risk_diff)) * np.sign(loss_to_desired_risk_diff)
        if self.score_func is not None:
            score = float(self.score_func.score(y, calibrated_pred_sets))
            d *= score
        self.theta_tag = np.clip(self.theta_tag + d, self.low, self.high)

    @classmethod
    def get_params_space(cls):
        return [
            UniformFloatHyperparameter("beta1", lower=0.01, upper=0.4),
            UniformFloatHyperparameter("beta2", lower=0.1, upper=2.0),
        ]

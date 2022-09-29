from typing import Optional

import numpy as np
from ConfigSpace import UniformFloatHyperparameter

from utils.Calibration.PhiRiskCalibration import PhiFunction, ExpPhiFunction, IdentityPhiFunction
from utils.Calibration.ScoreFunction import ScoreFunction


class ScoreAwarePhiFunction(PhiFunction):
    def __init__(self, change_mean, beta1, score_func: Optional[ScoreFunction] = None, with_exp_phi=False, **kwargs):
        super().__init__(**kwargs)
        self.low = -change_mean
        self.high = change_mean
        self.beta1 = beta1
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
        d = self.beta1 * np.sign(loss_to_desired_risk_diff)
        score = float(self.score_func.score(y, calibrated_pred_sets))
        d *= score
        self.theta_tag = np.clip(self.theta_tag + d, self.low, self.high)

    @classmethod
    def get_params_space(cls):
        return [
            UniformFloatHyperparameter("beta1", lower=0.01, upper=0.4),
        ]

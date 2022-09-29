import numpy as np
import torch

from utils.Calibration.CoverageRiskCalibration import CoverageOutputToPredSet
from utils.Calibration.RiskCalibration import RiskLoss
from utils.Calibration.RiskCalibrationBase import RiskCalibrationBase


class DummyRiskLoss(RiskLoss):
    def calc(self, y, y_tag, y_history, y_tag_history) -> torch.FloatTensor:
        if y_tag[-1, 1] - y_tag[-1, 0] <= 0:
            ret = np.e ** (2 * torch.ones(y_tag.shape[0]))  # max risk
        elif y_tag[-1, 1] - y_tag[-1, 0] > 1e8:
            ret = torch.ones(y_tag.shape[0])  # no risk
        else:
            ret = np.e**(2 * torch.rand(y_tag.shape[0]))
        return ret.float().unsqueeze(-1)


class DummyRiskCalibration(RiskCalibrationBase):
    def __init__(self, desired_msl_level, gamma, phi, calibration_size=1500, **kwargs):
        super().__init__(desired_risk_level=desired_msl_level, gamma=gamma, phi=phi,
                         risk_prefix='DUMMY_RISK_', risks_loss_history_size=40, tau=CoverageOutputToPredSet(),
                         loss=DummyRiskLoss(), calibration_size=calibration_size, **kwargs)

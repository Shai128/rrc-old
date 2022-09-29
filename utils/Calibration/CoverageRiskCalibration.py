import torch

from utils.Calibration.ACICalibration import ACICalibration
from utils.Calibration.CoverageRiskLoss import CoverageRiskLoss
from utils.Calibration.LossAwarePhiFunction import LossAwarePhiFunction
from utils.Calibration.PhiRiskCalibration import (PhiRiskCalibration, OutputToPredSet, IdentityPhiFunction,
                                                  ExpPhiFunction)
from utils.Calibration.RiskCalibration import SingleScalarOutputToPredSetModel
from utils.Calibration.RiskCalibrationBase import RiskCalibrationBase
from utils.Model import Model


class CoverageOutputToPredSet(OutputToPredSet):
    def to_pred_set(self, model_outputs, params: float):
        """
        Args:
            model_outputs (torch.FloatTensor): model outputs tensor of shape (batch_size x 2) or (batch_size x seq_len x 2)
        Returns:
            A calibrated intervals tensor of shape (batch_size x 2) or (batch_size x seq_len x 2)
        """
        calibrated = model_outputs.clone()
        if len(calibrated.shape) == 2:
            calibrated[:, 0] -= params
            calibrated[:, 1] += params
        else:
            calibrated[:, :, 0] -= params
            calibrated[:, :, 1] += params
        return calibrated


class CoverageOutputToPredSetModel(SingleScalarOutputToPredSetModel):
    def to_pred_set(self, model_outputs):
        """
        Args:
            model_outputs (torch.FloatTensor): model outputs tensor of shape (batch_size x 2) or (batch_size x seq_len x 2)
        Returns:
            A calibrated intervals tensor of shape (batch_size x 2) or (batch_size x seq_len x 2)
        """
        return CoverageOutputToPredSet().to_pred_set(model_outputs, self.theta)


class RCIUsingRiskCalibration(RiskCalibrationBase):
    def __init__(self, desired_coverage_level, gamma, phi, calibration_size=1500, model: Model = None, **kwargs):
        super().__init__(desired_risk_level=1-desired_coverage_level, gamma=gamma, phi=phi,
                         risk_prefix='COVERAGE_RISK_', risks_loss_history_size=0, tau=CoverageOutputToPredSet(), loss=CoverageRiskLoss(),
                         calibration_size=calibration_size, **kwargs)
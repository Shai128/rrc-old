from utils.Calibration.CoverageRiskCalibration import CoverageOutputToPredSet
from utils.Calibration.Msl2RiskLoss import Msl2RiskLoss
from utils.Calibration.RiskCalibrationBase import RiskCalibrationBase
from utils.Model import Model


class Msl2RiskCalibration(RiskCalibrationBase):
    def __init__(self, desired_msl_level, gamma, phi, calibration_size=1500, model: Model = None, **kwargs):
        super().__init__(desired_risk_level=desired_msl_level, gamma=gamma, phi=phi,
                         risk_prefix='MSL_RISK_', risks_loss_history_size=40, tau=CoverageOutputToPredSet(), loss=Msl2RiskLoss(),
                         calibration_size=calibration_size, **kwargs)

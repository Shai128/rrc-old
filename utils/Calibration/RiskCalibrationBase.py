import torch

from utils.Calibration.ACICalibration import ACICalibration
from utils.Calibration.LossAwarePhiFunction import LossAwarePhiFunction
from utils.Calibration.PhiRiskCalibration import (PhiRiskCalibration, IdentityPhiFunction, ExpPhiFunction,
                                                  OutputToPredSet)
from utils.Calibration.RiskCalibration import RiskLoss
from utils.Calibration.ScoreAwarePhiFunction import ScoreAwarePhiFunction
from utils.Calibration.ScoreFunction import CoverageScoreFunction
from utils.Model import Model
from utils.UncertaintyQuantificationResults.PredictionIntervals import PredictionIntervals


class RiskCalibrationBase(ACICalibration):
    def __init__(self, desired_risk_level, gamma, phi, tau: OutputToPredSet, loss: RiskLoss, risk_prefix: str,
                 risks_loss_history_size: int, calibration_size=1500, model: Model = None, search_params=True, phi_params=None, **kwargs):
        super().__init__(desired_coverage_level=desired_risk_level, gamma=gamma, calibration_size=calibration_size,
                         **kwargs)
        self.Q_t = 0
        self.alpha = desired_risk_level
        self.calibration_method = risk_prefix + phi
        self.previous_Q_t = []
        self.model = model

        if phi_params is None:
            phi_params = {}

        for key, value in phi_params.items():
            self.calibration_method += f'_{key}_{value}'

        update_phi_after = kwargs['train_len'] - kwargs['calibration_starting_update_index']

        tmp_phi_params = phi_params

        if phi == 'IdentityPhiFunction':
            phi_type = IdentityPhiFunction
        elif phi == 'ExpPhiFunction':
            phi_type = ExpPhiFunction
        elif phi == 'LossAwarePhiFunction':
            phi_type = LossAwarePhiFunction
        elif phi == 'LossAwarePhiFunction_CoverageScoreFunction':
            phi_type = LossAwarePhiFunction
            phi_params = dict(score_func=CoverageScoreFunction())
        elif phi == 'LossAwareExpPhiFunction_CoverageScoreFunction':
            phi_type = LossAwarePhiFunction
            phi_params = dict(score_func=CoverageScoreFunction(), with_exp_phi=True)
        elif phi == 'ScoreAwarePhiFunction':
            phi_type = ScoreAwarePhiFunction
            phi_params = dict(score_func=CoverageScoreFunction())
        elif phi == 'ScoreAwareExpPhiFunction':
            phi_type = ScoreAwarePhiFunction
            phi_params = dict(score_func=CoverageScoreFunction(), with_exp_phi=True)
        else:
            raise ValueError('Unknown phi function')
        phi_params.update(tmp_phi_params)

        self.risk_calibration = PhiRiskCalibration(
            desired_risk=self.alpha,
            tau=tau,
            risks_losses=loss,
            phi_type=phi_type,
            phi_params=phi_params,
            update_phi_after=update_phi_after,
            gamma=gamma,
            risks_loss_history_size=risks_loss_history_size,
            search_params=search_params,
        )

    def fit(self, **kwargs):
        pass

    def calibrate(self, x, y, predicted_interval: PredictionIntervals, **kwargs):
        predicted_interval = predicted_interval.intervals
        if len(predicted_interval.shape) == 1:
            predicted_interval = predicted_interval.unsqueeze(0)

        calibrated_interval = self.risk_calibration.to_pred_set(model_outputs=predicted_interval)
        calibrated_interval = PredictionIntervals(calibrated_interval)
        return calibrated_interval

    def update(self, x_t, y_t, uncalibrated_interval_t, calibrated_interval_t, uncalibrated_interval_history, **kwargs):
        y_t = y_t.squeeze(0)[-1]  # take only the new y
        uncalibrated_interval_history = uncalibrated_interval_history.squeeze(0)[-1]  # take only the new predicted interval
        assert len(y_t.shape) == 1
        assert len(uncalibrated_interval_history.shape) == 1
        self.risk_calibration.update(y_t, uncalibrated_interval_history)
        self.previous_Q_t += [self.Q_t]

    def plot_parameter_vs_time(self, starting_time, ending_time, save_dir=None):
        if self.model is not None:
            inverse_transformed_Q = self.model.scaler.unscale_y(torch.Tensor(self.previous_Q_t).unsqueeze(-1))
            ACICalibration.plot_parameter_vs_time_aux(self, inverse_transformed_Q, starting_time, ending_time,
                                                      'Q', save_dir=save_dir)

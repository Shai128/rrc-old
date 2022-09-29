import torch

from utils.Calibration.RiskCalibration import RiskLoss


class CoverageRiskLoss(RiskLoss):
    def calc(self, y, y_tag, y_history=None, y_tag_history=None) -> torch.FloatTensor:
        """
        Args:
            y (torch.FloatTensor): labels tensor of shape (batch_size x 1)
            y_tag: prediction sets tensor of shape (batch_size x 2)

        Returns:
            0 if `y` is in `y_tag`, else 1 (for each element in the batch).
        """
        assert len(y.shape) == 2
        assert len(y_tag.shape) == 2
        interval = y_tag
        res = 1.0 - ((y <= interval[:, 1].unsqueeze(1)) & (y >= interval[:, 0].unsqueeze(1))).float()
        return res
import torch

from utils.Calibration.RiskCalibration import RiskLoss


def get_streak_lengths(y, upper_q, lower_q):
    coverages = (y <= upper_q) & (y >= lower_q)

    if len(coverages.shape) == 1:
        coverages = coverages.unsqueeze(0)
    assert len(coverages.shape) == 2
    v = 1 - coverages.int()
    streaks_lengths = torch.zeros_like(v)
    streaks_lengths[:, -1] = v[:, -1]

    for t in range(v.shape[-1]-1):
        streaks_lengths[:, t+1] = (streaks_lengths[:, t] + v[:, t+1])*v[:, t+1]
    return streaks_lengths


class Msl2RiskLoss(RiskLoss):
    def calc(self, y, y_tag, y_history, y_tag_history, last_only=True) -> torch.FloatTensor:
        """
        Args:
            y (torch.FloatTensor): labels tensor of shape (batch_size x 1)
            y_tag: prediction sets tensor of shape (batch_size x 2)
            y_history: prediction sets tensor of shape (batch_size x window_size x 2)
            y_tag_history: prediction sets tensor of shape (batch_size x window_size x 2)

        Returns:
            returns the MSL (for each window in the batch).
        """
        assert y_history is not None
        assert y_tag_history is not None
        assert len(y_history.shape) == 3
        assert len(y_tag_history.shape) == 3
        losses = torch.zeros(y_history.shape[0], 1)

        y_history = y_history[:, -y_tag_history.shape[1]:, :]

        res = get_streak_lengths(
            y_history.squeeze(-1),
            y_tag_history[:, :, 1].squeeze(-1),
            y_tag_history[:, :, 0].squeeze(-1),
        )

        if last_only:
            res = res[:, -1]
        else:
            res = res.float().mean(dim=-1)

        if type(res) is float:
            assert torch.numel(losses) == 1
        losses[:, 0] = res

        return losses

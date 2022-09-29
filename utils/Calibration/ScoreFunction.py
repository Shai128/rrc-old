import abc
import torch


class ScoreFunction(abc.ABC):
    @abc.abstractmethod
    def score(self, y, pred_sets) -> torch.FloatTensor:
        raise NotImplementedError()


class CoverageScoreFunction(ScoreFunction):
    def score(self, y, pred_sets) -> torch.FloatTensor:
        assert len(y.shape) == len(pred_sets.shape)
        diff_from_q_high = torch.abs(pred_sets[..., 1].unsqueeze(-1) - y)
        diff_from_q_low = torch.abs(pred_sets[..., 0].unsqueeze(-1) - y)
        diff_from_closest_q = torch.min(torch.stack([diff_from_q_high, diff_from_q_low], dim=-1), dim=-1).values
        return diff_from_closest_q.float()

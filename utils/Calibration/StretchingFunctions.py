import abc

import numpy as np
import torch


def exp(x: torch.Tensor, base) -> torch.Tensor:
    device = x.device
    x = x.cpu().numpy()
    return torch.tensor((np.nan_to_num(np.power(base, x) - 1)) * (x > 0) -
                        (np.nan_to_num(np.power(base, -x) - 1)) * (x < 0)).to(device)


class StretchingFunction(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def __call__(self, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def update(self, theta: torch.Tensor, **kwargs) -> None:
        pass


class IdentityStretching(StretchingFunction):

    def __init__(self):
        super().__init__()

    def name(self):
        return "identity"

    def __call__(self, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        return theta


class ExponentialStretching(StretchingFunction):

    def __init__(self, base=np.e):
        super().__init__()
        self.base = base

    def name(self):
        base_name = 'e' if self.base == np.e else np.round(self.base, 3)
        return f"exp_{base_name}"

    def __call__(self, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        base = self.base
        linear_idx = (-0.1 <= exp(theta, base)) & (exp(theta, base) <= 0.1)
        return linear_idx * theta + (~linear_idx) * exp(theta, base)


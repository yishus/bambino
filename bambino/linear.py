import math
import torch

type DeviceLikeType = str | torch.device | int


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = torch.empty(
            (out_features, in_features), device=device, dtype=dtype
        )

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x @ self.weight.T

        return output

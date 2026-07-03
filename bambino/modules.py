import math
import torch
import torch.nn as nn

type DeviceLikeType = str | torch.device | int


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # By storing the weight matrix as (out_features, in_features),
        # the gradient matrix perfectly matches the layout of the weight matrix natively.
        self.weight = torch.empty(
            (out_features, in_features), device=device, dtype=dtype
        )

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x @ self.weight.T

        return output


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = torch.empty(
            (num_embeddings, embedding_dim), device=device, dtype=dtype
        )

        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.weight[x]

        return output


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        result = x * rms * self.weight

        return result.to(in_dtype)

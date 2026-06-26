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
        # By storing the weight matrix as (out_features, in_features),
        # the gradient matrix perfectly matches the layout of the weight matrix natively.
        self.weight = torch.empty(
            (out_features, in_features), device=device, dtype=dtype
        )

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x @ self.weight.T

        return output


class Embedding(torch.nn.Module):
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

        torch.nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.weight[x]

        return output

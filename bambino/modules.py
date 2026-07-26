import math

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce, repeat

type DeviceLikeType = str | torch.device | int


class Linear(nn.Module):
    """A bias-free linear transformation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x,
            self.weight,
            "... in_features, out_features in_features -> ... out_features",
        )

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"


class Embedding(nn.Module):
    """A lookup table that stores embeddings for a fixed-size vocabulary."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("num_embeddings and embedding_dim must be positive")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}"
        )


class RMSNorm(nn.Module):
    """Root mean square normalization over the final dimension."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if eps < 0:
            raise ValueError("eps must be non-negative")

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError("RMSNorm expects a floating-point input")
        if x.ndim == 0 or x.shape[-1] != self.d_model:
            actual_size = x.shape[-1] if x.ndim > 0 else None
            raise ValueError(
                f"expected the final dimension to be {self.d_model}, got {actual_size}"
            )

        input_dtype = x.dtype
        compute_dtype = (
            torch.float32
            if input_dtype in (torch.float16, torch.bfloat16)
            else input_dtype
        )
        x_compute = x.to(compute_dtype)
        mean_square = reduce(x_compute.square(), "... d -> ... 1", "mean")
        inverse_rms = torch.rsqrt(mean_square + self.eps)
        return (x_compute * inverse_rms * self.weight).to(input_dtype)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"


def default_d_ff(d_model: int, multiple_of: int = 64) -> int:
    """Return the parameter-matched SwiGLU hidden size, rounded up for hardware."""
    if d_model <= 0:
        raise ValueError("d_model must be positive")
    if multiple_of <= 0:
        raise ValueError("multiple_of must be positive")

    # A gated FFN has three projection matrices, so 8/3 preserves roughly the
    # same parameter count as a conventional FFN with a hidden size of 4*d_model.
    raw_d_ff = 8 * d_model // 3
    return ((raw_d_ff + multiple_of - 1) // multiple_of) * multiple_of


class SwiGLUFFN(nn.Module):
    """Bias-free SwiGLU feed-forward network with a combined input projection."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")

        d_ff = default_d_ff(d_model) if d_ff is None else d_ff
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")

        self.d_model = d_model
        self.d_ff = d_ff
        # Combining the gate and value projections saves a matrix multiplication.
        self.w_gate_value = Linear(d_model, 2 * d_ff, device=device, dtype=dtype)
        self.w_down = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.w_gate_value(x).chunk(2, dim=-1)
        silu_gate = gate * torch.sigmoid(gate)
        return self.w_down(silu_gate * value)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}"


class RotaryPositionalEmbedding(nn.Module):
    """Apply half-rotation RoPE to tensors shaped ``(..., sequence, d_k)``."""

    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: DeviceLikeType | None = None,
    ) -> None:
        super().__init__()
        if theta <= 0:
            raise ValueError("theta must be positive")
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k must be a positive even integer")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        dimension_indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inverse_frequencies = theta ** (-dimension_indices / d_k)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        frequencies = einsum(positions, inverse_frequencies, "s, d -> s d")
        # Half-rotation RoPE pairs the first and second halves of the feature axis.
        angles = repeat(frequencies, "s d -> s (two d)", two=2)

        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError("RoPE expects a floating-point input")
        if x.ndim < 2:
            raise ValueError("x must have at least a sequence and feature dimension")
        if x.shape[-1] != self.d_k:
            raise ValueError(
                f"expected the final dimension to be {self.d_k}, got {x.shape[-1]}"
            )
        if token_positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_positions must contain 32-bit or 64-bit integers")
        if token_positions.ndim < 1 or token_positions.ndim >= x.ndim:
            raise ValueError(
                "token_positions must have between one and x.ndim - 1 dimensions"
            )
        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError(
                "the final dimension of token_positions must match x's sequence length"
            )

        cos = self.cos_cached[token_positions].to(dtype=x.dtype)
        sin = self.sin_cached[token_positions].to(dtype=x.dtype)

        # Insert singleton head/batch axes immediately before sequence as needed.
        while cos.ndim < x.ndim:
            cos = rearrange(cos, "... sequence d_k -> ... 1 sequence d_k")
            sin = rearrange(sin, "... sequence d_k -> ... 1 sequence d_k")

        first_half, second_half = x.chunk(2, dim=-1)
        rotated = torch.cat((-second_half, first_half), dim=-1)
        return x * cos + rotated * sin

    def extra_repr(self) -> str:
        return f"theta={self.theta}, d_k={self.d_k}, " f"max_seq_len={self.max_seq_len}"

import math
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

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

        variance = reduce(x**2, "b s d -> b s 1", "mean")
        rms = torch.rsqrt(variance + self.eps)

        result = x * rms * self.weight

        return result.to(in_dtype)


def default_d_ff(d_model: int, multiple_of: int = 64) -> int:
    # 1. Compute the raw 8/3 scaling
    raw_d_ff = int(8 / 3 * d_model)

    # 2. Round up to the nearest multiple of 64
    # (raw + 63) // 64 * 64 handles the ceiling division trick
    d_ff = ((raw_d_ff + multiple_of - 1) // multiple_of) * multiple_of

    return d_ff


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        dff = d_ff or default_d_ff(d_model)

        # Combine W1 (gate) and W3 (value) into a single projection matrix
        # Output dimension is exactly 2 * d_ff
        self.w_gate_value = Linear(d_model, dff * 2, device=device, dtype=dtype)

        # W2 (down-projection matrix)
        self.w_down = Linear(dff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        combined_proj = self.w_gate_value(x)

        gate, value = rearrange(
            combined_proj, "b s (split d_ff) -> split b s d_ff", split=2
        )

        silu_gate = gate * torch.sigmoid(gate)
        intermediate = silu_gate * value

        return self.w_down(intermediate)


class RotaryPositionalEmbedding(nn.Module):
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        assert d_k % 2 == 0, "Dimension d_k must be even for RoPE."

        # Compute inverse frequencies and position matrix
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k)
        )
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)

        # Generates positions and frequencies, then interleaves them cleanly
        freqs = rearrange(t, "s -> s 1") * rearrange(inv_freq, "d -> 1 d")
        emb = repeat(freqs, "s d -> s (two d)", two=2)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 1. Gather the cache values for the given positions
        # Output shape: [batch, seq_len, d_k]
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 2. Match x's dimension layout automatically by specifying the head dimension 'h'
        cos = rearrange(cos, "b s d -> b 1 s d")
        sin = rearrange(sin, "b s d -> b 1 s d")

        # 3. Perform the half-slice rotation trick in one declarative line
        # Splits the 'd' dimension into two halves (d1, d2), negates d2, and swaps them
        x_rotated = rearrange(
            x, "b h s (two d) -> b h s (d two)", two=2
        )  # step to separate pairs
        x1, x2 = x_rotated.chunk(2, dim=-1)
        x_rotated = torch.cat((-x2, x1), dim=-1)

        return (x * cos) + (x_rotated * sin)

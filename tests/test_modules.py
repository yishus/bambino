import pytest
import torch
import torch.nn.functional as F

from bambino.modules import (
    Embedding,
    Linear,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLUFFN,
    default_d_ff,
)


def test_linear_registers_weight_and_matches_functional_linear():
    layer = Linear(3, 5)
    x = torch.randn(2, 4, 3)

    torch.testing.assert_close(layer(x), F.linear(x, layer.weight))
    assert dict(layer.named_parameters())["weight"] is layer.weight


def test_embedding_registers_weight_and_matches_lookup():
    embedding = Embedding(7, 3)
    indices = torch.tensor([[0, 2], [4, 6]])

    torch.testing.assert_close(embedding(indices), embedding.weight[indices])
    assert dict(embedding.named_parameters())["weight"] is embedding.weight


def test_rms_norm_supports_arbitrary_leading_dimensions():
    norm = RMSNorm(4, eps=1e-5)
    x = torch.randn(2, 3, 5, 4)
    expected = F.rms_norm(x, (4,), norm.weight, norm.eps)

    torch.testing.assert_close(norm(x), expected)


def test_default_d_ff_rounds_up_to_requested_multiple():
    assert default_d_ff(128, multiple_of=64) == 384
    assert default_d_ff(128, multiple_of=256) == 512


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((0,), "d_model"),
        ((8, 0), "multiple_of"),
    ],
)
def test_default_d_ff_rejects_invalid_arguments(arguments, message):
    with pytest.raises(ValueError, match=message):
        default_d_ff(*arguments)


def test_swiglu_matches_explicit_formula_and_preserves_leading_dimensions():
    module = SwiGLUFFN(d_model=4, d_ff=6)
    x = torch.randn(2, 3, 5, 4)

    gate, value = F.linear(x, module.w_gate_value.weight).chunk(2, dim=-1)
    expected = F.linear(F.silu(gate) * value, module.w_down.weight)

    torch.testing.assert_close(module(x), expected)
    assert module(x).shape == x.shape


def test_rope_position_zero_is_identity():
    rope = RotaryPositionalEmbedding(theta=10_000.0, d_k=4, max_seq_len=8)
    x = torch.randn(2, 3, 1, 4)

    output = rope(x, torch.tensor([0]))

    torch.testing.assert_close(output, x)


def test_rope_supports_shared_and_per_batch_positions():
    rope = RotaryPositionalEmbedding(theta=10_000.0, d_k=4, max_seq_len=8)
    x = torch.randn(2, 3, 4, 4)
    shared_positions = torch.arange(4)
    batch_positions = shared_positions.expand(2, -1)

    shared_output = rope(x, shared_positions)
    batch_output = rope(x, batch_positions)

    torch.testing.assert_close(shared_output, batch_output)
    assert shared_output.dtype == x.dtype

"""Mergen-SNN — Cython AVX2 SIMD GEMM Birim Testleri."""

import pytest
import torch
import torch.nn.functional as F

from mergen.cython_ext.fast_gemm import fast_gemm_forward
from mergen.models.snn_lm_bpe import FastLinear


def test_fast_gemm_standard_matmul():
    """fast_gemm_forward standart torch.matmul ile allclose olmalıdır."""
    torch.manual_seed(42)
    M, K, N = 32, 64, 48
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)

    expected = torch.matmul(A, B)
    actual_np = fast_gemm_forward(A.numpy(), B.numpy(), transposed_b=False)
    actual = torch.from_numpy(actual_np)

    assert torch.allclose(expected, actual, atol=1e-4)


def test_fast_gemm_linear_with_bias():
    """fast_gemm_forward PyTorch F.linear (A @ weight.T + bias) ile allclose olmalıdır."""
    torch.manual_seed(42)
    M, K, N = 64, 128, 96
    A = torch.randn(M, K, dtype=torch.float32)
    weight = torch.randn(N, K, dtype=torch.float32)
    bias = torch.randn(N, dtype=torch.float32)

    expected = F.linear(A, weight, bias)
    actual_np = fast_gemm_forward(A.numpy(), weight.numpy(), bias.numpy(), transposed_b=True)
    actual = torch.from_numpy(actual_np)

    assert torch.allclose(expected, actual, atol=1e-4)


def test_fast_linear_module_forward_backward():
    """FastLinear modülü hem forward hem backward geçişte geçerli gradyanlar üretmelidir."""
    torch.manual_seed(42)
    in_features, out_features = 32, 64
    layer = FastLinear(in_features, out_features, bias=True)

    x = torch.randn(8, 16, in_features, requires_grad=True)
    out = layer(x)

    assert out.shape == (8, 16, out_features)
    assert not torch.isnan(out).any()

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(layer.weight.grad).any()

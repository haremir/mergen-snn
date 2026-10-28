"""Mergen-SNN — Cython LIF Hızlandırma Birim Testleri."""

import pytest
import snntorch as snn
import torch
import torch.nn.functional as F

from mergen.cython_ext.fast_lif import (
    fast_lif_forward,
    fast_lif_sequence_backward,
    fast_lif_sequence_forward,
)
from mergen.models.snn_lm_bpe import (
    BpeSNNLM,
    FastLeaky,
    FastLIFFunction,
    FastLIFSequence,
)


def test_cython_fast_lif_forward_kernel():
    """Doğrudan C++ Cython çekirdeğinin matematiksel doğruluğu."""
    x = torch.tensor([[0.5, 1.2], [-0.2, 2.0]], dtype=torch.float32)
    mem = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    spk, new_mem = fast_lif_forward(x.numpy(), mem.numpy(), 0.9, 1.0)

    # (0, 0): m = 0.5 <= 1.0 -> spk=0, mem=0.5
    assert spk[0, 0] == 0.0
    assert pytest.approx(new_mem[0, 0], 1e-5) == 0.5

    # (0, 1): m = 1.2 > 1.0 -> spk=1, mem=0.2
    assert spk[0, 1] == 1.0
    assert pytest.approx(new_mem[0, 1], 1e-5) == 0.2

    # (1, 0): m = -0.2 <= 1.0 -> spk=0, mem=-0.2
    assert spk[1, 0] == 0.0
    assert pytest.approx(new_mem[1, 0], 1e-5) == -0.2

    # (1, 1): m = 2.0 > 1.0 -> spk=1, mem=1.0
    assert spk[1, 1] == 1.0
    assert pytest.approx(new_mem[1, 1], 1e-5) == 1.0


def test_fast_leaky_autograd():
    """FastLeaky modülünün PyTorch autograd ile geriye doğru türev akışı."""
    layer = FastLeaky(beta=0.9, threshold=1.0)
    x = torch.randn(4, 16, requires_grad=True)
    mem = layer.init_leaky()

    spk, new_mem = layer(x, mem)
    assert spk.shape == (4, 16)
    assert new_mem.shape == (4, 16)

    loss = (spk * 2.0 + new_mem).sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert (x.grad != 0).any()


def test_fast_leaky_vs_snntorch_equivalence():
    """FastLeaky ile snnTorch Leaky aynı girdi için aynı spike çıktılarını vermeli."""
    fast = FastLeaky(beta=0.9, threshold=1.0)
    slow = snn.Leaky(beta=0.9, threshold=1.0, reset_mechanism="subtract")

    x = torch.tensor([[0.2, 1.5, 0.8, 1.1]], dtype=torch.float32)
    mem_fast = fast.init_leaky()
    mem_slow = slow.init_leaky()

    spk_fast, _ = fast(x, mem_fast)
    spk_slow, _ = slow(x, mem_slow)

    assert torch.equal(spk_fast, spk_slow)


def test_sequence_lif_vs_loop_equivalence():
    """FastLIFSequence hem ileri (spikes) hem geri (gradients) geçişte döngü modeliyle tam örtüşmeli."""
    B, T, H = 4, 16, 8
    torch.manual_seed(42)
    x1 = torch.randn(B, T, H, requires_grad=True)
    mem_init1 = torch.randn(B, H, requires_grad=True)

    torch.manual_seed(42)
    x2 = torch.randn(B, T, H, requires_grad=True)
    mem_init2 = torch.randn(B, H, requires_grad=True)

    # 1. Döngülü model (FastLeaky step-by-step)
    layer = FastLeaky(beta=0.9, threshold=1.0)
    mem = mem_init1
    spk_list = []
    for t in range(T):
        spk, mem = layer(x1[:, t, :].contiguous(), mem)
        spk_list.append(spk)
    spk_loop = torch.stack(spk_list, dim=1)
    loss_loop = (spk_loop * 2.5).sum() + mem.sum() * 0.3
    loss_loop.backward()

    # 2. Sequence model (Cython tek çağrı)
    spk_seq, mem_last_seq = FastLIFSequence.apply(x2, mem_init2, 0.9, 1.0)
    loss_seq = (spk_seq * 2.5).sum() + mem_last_seq.sum() * 0.3
    loss_seq.backward()

    # İleri ve geri geçiş doğrulaması
    assert torch.equal(spk_loop, spk_seq)
    assert torch.allclose(mem, mem_last_seq, atol=1e-5)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(mem_init1.grad, mem_init2.grad, atol=1e-5)


def test_bpe_snn_lm_modes():
    """BpeSNNLM hem use_cython=True hem de use_cython=False ile sorunsuz çalışmalıdır."""
    m_cython = BpeSNNLM(vocab_size=64, embed_dim=16, hidden_size=32, use_cython=True)
    m_snntorch = BpeSNNLM(vocab_size=64, embed_dim=16, hidden_size=32, use_cython=False)

    x = torch.randint(0, 64, (2, 8))
    out_cy = m_cython(x, num_steps=8)
    out_snn = m_snntorch(x, num_steps=8)

    assert out_cy.shape == (2, 8, 64)
    assert out_snn.shape == (2, 8, 64)

    # Geriye yayılım testi
    loss_cy = out_cy.sum()
    loss_cy.backward()
    assert m_cython.fc_out.weight.grad is not None

    loss_snn = out_snn.sum()
    loss_snn.backward()
    assert m_snntorch.fc_out.weight.grad is not None

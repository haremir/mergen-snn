"""Mergen-SNN — Stage 2 CharSNNLM Model Testleri."""

import torch
import torch.nn.functional as F

from mergen.models.snn_lm import CharSNNLM
from mergen.models.snn_lm_bpe import BpeSNNLM


def test_snn_lm_forward():
    model = CharSNNLM(vocab_size=100, embed_dim=16, hidden_size=32)
    x = torch.randint(0, 100, (2, 32))  # (batch=2, seq=32)
    logits = model(x, num_steps=32)
    assert logits.shape == (2, 32, 100)
    assert not torch.isnan(logits).any()


def test_snn_lm_backward():
    model = CharSNNLM(vocab_size=100, embed_dim=16, hidden_size=32)
    x = torch.randint(0, 100, (2, 32))
    y = torch.randint(0, 100, (2, 32))
    logits = model(x, num_steps=32)
    loss = F.cross_entropy(logits.reshape(-1, 100), y.reshape(-1))
    loss.backward()
    assert model.fc_out.weight.grad is not None
    assert not torch.isnan(model.fc_out.weight.grad).any()


def test_bpe_snn_lm_forward_backward():
    model = BpeSNNLM(vocab_size=128, embed_dim=32, hidden_size=64)
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    logits = model(x, num_steps=16)
    assert logits.shape == (2, 16, 128)
    assert not torch.isnan(logits).any()

    loss = F.cross_entropy(logits.reshape(-1, 128), y.reshape(-1))
    loss.backward()
    assert model.fc_out.weight.grad is not None
    assert not torch.isnan(model.fc_out.weight.grad).any()


def test_bpe_snn_lm_weight_tying():
    # embed_dim == hidden_size olduğunda weight tying aktif olmalı
    model = BpeSNNLM(vocab_size=256, embed_dim=64, hidden_size=64, tie_weights=True)
    assert model.tie_weights is True
    assert model.fc_out.weight is model.embedding.weight
    assert model.fc_out.bias is None

    # Parametre sayısı: embedding (256*64) + fc1 (64*64+64) + fc2 (64*64+64) = 16384 + 4160 + 4160 = 24704
    unique_params = sum(p.numel() for p in model.parameters())
    assert unique_params == 256 * 64 + (64 * 64 + 64) + (64 * 64 + 64)

    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
    logits = model(x, num_steps=8)
    assert logits.shape == (2, 8, 256)

    loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
    loss.backward()
    assert model.embedding.weight.grad is not None
    assert model.fc_out.weight.grad is not None
    assert model.embedding.weight.grad is model.fc_out.weight.grad

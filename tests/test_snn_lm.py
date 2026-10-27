"""Mergen-SNN — Stage 2 CharSNNLM Model Testleri."""

import torch
import torch.nn.functional as F

from mergen.models.snn_lm import CharSNNLM


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

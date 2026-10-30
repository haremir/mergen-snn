"""Unit tests for SFT data preparation, token expansion, and chat inference."""

import numpy as np
import pytest
import torch

from mergen.models.snn_lm_bpe import BpeSNNLM


def test_resize_token_embeddings():
    model = BpeSNNLM(vocab_size=8192, embed_dim=384, hidden_size=384, tie_weights=False)
    assert model.embedding.weight.shape == (8192, 384)
    assert model.fc_out.weight.shape == (8192, 384)

    model.resize_token_embeddings(8196)
    assert model.vocab_size == 8196
    assert model.embedding.weight.shape == (8196, 384)
    assert model.fc_out.weight.shape == (8196, 384)
    assert model.fc_out.bias.shape == (8196,)


def test_generate_with_metrics():
    model = BpeSNNLM(vocab_size=8196, embed_dim=384, hidden_size=384, tie_weights=False)
    prompt = [8192, 10, 20, 8195, 8193, 30, 8195, 8194]

    output, metrics = model.generate_with_metrics(
        prompt,
        max_new_tokens=10,
        temperature=0.8,
        end_token_id=8195,
        block_size=32,
    )

    assert isinstance(output, list)
    assert len(output) <= 10
    assert "loss" in metrics
    assert "ppl" in metrics
    assert "bpb" in metrics
    assert "spike_ratio" in metrics
    assert "confidence" in metrics
    assert "n_tokens" in metrics
    assert metrics["n_tokens"] == len(output)

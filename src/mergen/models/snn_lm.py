"""Mergen-SNN — Stage 2 Karakter Seviyesinde SNN Dil Modeli."""

import snntorch as snn
import torch
import torch.nn as nn


class CharSNNLM(nn.Module):
    """Karakter seviyesinde SNN Dil Modeli (Stage 2).

    Her timestep bir karakter; LIF membran potansiyeli zamanla taşınır → bağlam öğrenimi.
    Embedding çıktısı LIF'e current olarak verilir (spike değil).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 256,
        beta: float = 0.9,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.beta = beta
        self.threshold = threshold
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="none")
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="none")
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, num_steps: int = 32) -> torch.Tensor:
        # input_ids: (batch, num_steps) — int64
        batch_size = input_ids.size(0)
        emb = self.embedding(input_ids)  # (batch, num_steps, embed_dim)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        logits_list = []
        for t in range(num_steps):
            x_t = emb[:, t, :]  # (batch, embed_dim) — LIF'e current
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            mem1 = torch.where(spk1 > 0, mem1 - 0.5 * self.threshold, mem1)
            mem1 = torch.clamp(mem1, min=-0.5)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            mem2 = torch.where(spk2 > 0, mem2 - 0.5 * self.threshold, mem2)
            mem2 = torch.clamp(mem2, min=-0.5)
            logits_t = self.fc_out(spk2)  # (batch, vocab_size)
            logits_list.append(logits_t)
        return torch.stack(logits_list, dim=1)  # (batch, num_steps, vocab_size)

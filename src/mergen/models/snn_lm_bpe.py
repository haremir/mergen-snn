"""Mergen-SNN — Stage 3: BPE Tabanlı SNN Dil Modeli (Mergen-Pico, ~14M Parametre)."""

import snntorch as snn
import torch
import torch.nn as nn


class BpeSNNLM(nn.Module):
    """BPE tabanlı Spiking Sinir Ağı Dil Modeli (Mergen-Pico).

    Her timestep bir BPE alt-kelimesi (token) işler.
    LIF membran potansiyelleri zaman boyunca taşınarak bağlam korunur.
    Embedding çıkışı FC1 ve FC2 katmanları ile LIF nöronları üzerinden iletilir.
    Matris çarpımları döngü dışına çıkarılarak paralel CPU GEMM performansı maksimize edilmiştir.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        embed_dim: int = 512,
        hidden_size: int = 1024,
        beta: float = 0.9,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, num_steps: int = 128) -> torch.Tensor:
        # input_ids: (batch, seq_len)
        batch_size = input_ids.size(0)
        emb = self.embedding(input_ids)  # (batch, seq_len, embed_dim)

        # FC1 projeksiyonu tüm dizi için paralel hesaplanır
        cur1_all = self.fc1(emb)  # (batch, seq_len, hidden_size)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_list = []

        # Ardışık zaman adımlı LIF membran entegrasyonu
        for t in range(num_steps):
            cur1_t = cur1_all[:, t, :]
            spk1, mem1 = self.lif1(cur1_t, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_list.append(spk2)

        # Çıkış projeksiyonu (FC_out) tüm adımlar için tek bir GEMM matmul ile hesaplanır
        spk2_all = torch.stack(spk2_list, dim=1)  # (batch, seq_len, hidden_size)
        return self.fc_out(spk2_all)  # (batch, seq_len, vocab_size)

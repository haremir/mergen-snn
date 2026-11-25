"""Mergen-SNN — Spiking Transformer Dil Modeli (Spikformer / SpikeGPT Mimarisi).

Spiking Self-Attention (SSA) ve Spiking FFN blokları ile inşa edilmiş,
olay güdümlü (event-driven) darbe tabanlı derin dönüştürücü dil modeli.
"""

from typing import Dict, Optional, Tuple, Union

import snntorch as snn
import torch
import torch.nn as nn


class SpikingSelfAttention(nn.Module):
    """Spiking Self-Attention (SSA).

    Q, K, V spike olarak üretilir; softmax yerine normalize(Q @ K^T) kullanılır.
    Aktivasyonlar binary spike olduğu için çarpma işlemleri sinaptik toplamaya (AC) indirgenir.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        beta: float = 0.9,
        threshold: float = 0.5,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.lif_q = snn.Leaky(beta=beta, threshold=threshold)
        self.lif_k = snn.Leaky(beta=beta, threshold=threshold)
        self.lif_v = snn.Leaky(beta=beta, threshold=threshold)
        self.lif_o = snn.Leaky(beta=beta, threshold=threshold)

        self.register_buffer("last_spk_rate", torch.tensor(0.0), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        mem_states: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: (B, T, D)
        B, T, D = x.shape

        q, mem_q = self.lif_q(self.wq(x), mem_states["q"])
        k, mem_k = self.lif_k(self.wk(x), mem_states["k"])
        v, mem_v = self.lif_v(self.wv(x), mem_states["v"])

        # Multi-head reshape: (B, H, T, head_dim)
        q_heads = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_heads = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_heads = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: spike Q @ spike K^T, normalize (softmax yok, spike binary)
        attn = torch.matmul(q_heads, k_heads.transpose(-2, -1))  # (B, H, T, T)
        attn = attn / (self.head_dim ** 0.5)

        # Causal mask (Gelecek token'ları maskele)
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn = attn.masked_fill(~mask, 0.0)

        # Basit normalizasyon (spike dağılımı için)
        denom = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attn = attn / denom

        out = torch.matmul(attn, v_heads)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        out, mem_o = self.lif_o(self.wo(out), mem_states["o"])
        new_mem = {"q": mem_q, "k": mem_k, "v": mem_v, "o": mem_o}

        # Spike firing rate takibi
        spk_q = (q > 0).float().mean()
        spk_k = (k > 0).float().mean()
        spk_v = (v > 0).float().mean()
        spk_o = (out > 0).float().mean()
        self.last_spk_rate = ((spk_q + spk_k + spk_v + spk_o) / 4.0).detach()

        return out, new_mem

    def init_mems(self, batch: int, device: torch.device) -> Dict[str, torch.Tensor]:
        return {n: getattr(self, f"lif_{n}").init_leaky() for n in ["q", "k", "v", "o"]}


class SpikingFFN(nn.Module):
    """Spiking Feed-Forward Network (SwiGLU tarzı gating + LIF nöronlar)."""

    def __init__(
        self,
        dim: int,
        hidden: int,
        beta: float = 0.9,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold)

        self.register_buffer("last_spk_rate", torch.tensor(0.0), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        mems: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        a, m1 = self.lif1(self.w1(x), mems["1"])
        b, m2 = self.lif2(self.w2(x), mems["2"])
        h = a * b  # Spiking gating (ikili çarpım / AND)
        out, m3 = self.lif3(self.w3(h), mems["3"])

        spk_a = (a > 0).float().mean()
        spk_b = (b > 0).float().mean()
        spk_out = (out > 0).float().mean()
        self.last_spk_rate = ((spk_a + spk_b + spk_out) / 3.0).detach()

        return out, {"1": m1, "2": m2, "3": m3}

    def init_mems(self, batch: int, device: torch.device) -> Dict[str, torch.Tensor]:
        return {n: getattr(self, f"lif{n}").init_leaky() for n in ["1", "2", "3"]}


class SpikeTransformerBlock(nn.Module):
    """Pre-LayerNorm ve Residual bağlantılı Spiking Transformer Bloğu."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ffn_hidden: Optional[int] = None,
        beta: float = 0.9,
        threshold: float = 0.5,
    ):
        super().__init__()
        ffn_hidden = ffn_hidden or dim * 2
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpikingSelfAttention(dim, num_heads, beta, threshold)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SpikingFFN(dim, ffn_hidden, beta, threshold)

    def forward(
        self,
        x: torch.Tensor,
        mems: Dict[str, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        a, new_m1 = self.attn(self.norm1(x), mems["attn"])
        x = x + a
        f, new_m2 = self.ffn(self.norm2(x), mems["ffn"])
        x = x + f
        return x, {"attn": new_m1, "ffn": new_m2}

    def init_mems(self, batch: int, device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "attn": self.attn.init_mems(batch, device),
            "ffn": self.ffn.init_mems(batch, device),
        }


class SpikeTransformerLM(nn.Module):
    """Spiking Transformer Dil Modeli (10M Prototip / Spikformer).

    SSA (Spiking Self-Attention) ve LIF nöronları ile donatılmış,
    uzun bağlam yakalama yeteneğine sahip nöromorfik transformer.
    """

    def __init__(
        self,
        vocab_size: int = 16384,
        embed_dim: int = 256,
        num_blocks: int = 4,
        num_heads: int = 4,
        ffn_hidden: int = 512,
        max_seq: int = 128,
        beta: float = 0.9,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = embed_dim  # trainer uyumluluğu için
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.ffn_hidden = ffn_hidden
        self.max_seq = max_seq
        self.beta = beta
        self.threshold = threshold
        self.is_spike_transformer = True

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(max_seq, embed_dim)  # Öğrenilebilir pozisyonel gömme
        self.blocks = nn.ModuleList([
            SpikeTransformerBlock(embed_dim, num_heads, ffn_hidden, beta, threshold)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.register_buffer("last_spk_rate", torch.tensor(0.0), persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        num_steps: Optional[int] = None,
        return_spikes: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T = input_ids.shape
        T = min(T, self.max_seq)
        input_ids = input_ids[:, :T]

        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.embed(input_ids) + self.pos(pos_ids)

        # Her blok için membran potansiyellerini başlat
        all_mems = [blk.init_mems(B, input_ids.device) for blk in self.blocks]

        for i, blk in enumerate(self.blocks):
            x, all_mems[i] = blk(x, all_mems[i])

        x = self.norm(x)
        logits = self.lm_head(x)

        # Attention ve genel spike oranını takip et
        spk_rate = torch.stack([blk.attn.last_spk_rate for blk in self.blocks]).mean().detach()
        self.last_spk_rate = spk_rate

        if return_spikes:
            return logits, spk_rate
        return logits, spk_rate

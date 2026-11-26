"""Mergen-SNN — Hibrit LIF + Attention Dil Modeli (Hybrid LIF-Transformer).

Mimari Özellikleri:
- Q ve K: Doğrudan float Linear projeksiyonlar (LIF yok).
- V: LIF nöronu üzerinden üretilen darbe dizisi (binary spike {0, 1}).
- Attention: Softmax(Q @ K^T / sqrt(d)) @ V_spike (Float attention haritası, darbe tabanlı değerler).
- FFN: Spiking SwiGLU FFN (LIF nöronları ile darbe tabanlı gating).
- Saf SSA'nın binary bilgi darboğazını çözerek sürekli dikkat hassasiyetini ve darbe verimliliğini birleştirir.
"""

from typing import Dict, Optional, Tuple, Union

import snntorch as snn
import torch
import torch.nn as nn


class HybridSelfAttention(nn.Module):
    """Hibrit Self-Attention (Float Q, K + Spike V).

    Q ve K sürekli (float) temsil ile ince anlamsal ilişkileri ve dikkat dağılımını korur.
    V ise LIF nöronu ile darbe (spike) olarak üretilir.
    Darbe tabanlı V, sinaptik bellek ve olay güdümlü dinamikleri korurken
    saf SSA'daki binary dot-product bilgi kaybını ortadan kaldırır.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        beta: float = 0.9,
        threshold: float = 0.5,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Q ve K doğrudan float doğrusal projeksiyonlar (LIF yok)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)

        # V için doğrusal projeksiyon + LIF nöronu (Spike üretimi)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.lif_v = snn.Leaky(beta=beta, threshold=threshold)

        # Çıkış doğrusal projeksiyonu (Float projeksiyon)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.register_buffer("last_spk_rate", torch.tensor(0.0), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        mem_states: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: (B, T, D)
        B, T, D = x.shape

        # 1. Q ve K doğrudan float olarak üretilir
        q = self.wq(x)
        k = self.wk(x)

        # 2. V darbe (spike) olarak LIF nöronundan geçirilir
        v, mem_v = self.lif_v(self.wv(x), mem_states["v"])

        # Çok başlı (Multi-head) tensör şekillendirme: (B, H, T, head_dim)
        q_heads = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_heads = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_heads = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Sürekli Attention Ağırlıkları (Softmax ile causal maskeli)
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Gelecek token'ları maskele (Causal autoregressive mask)
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)

        # 4. Dikkat ağırlıkları ile spike V çarpımı: Softmax(Q @ K^T) @ V_spike
        out = torch.matmul(attn_weights, v_heads)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # 5. Çıkış projeksiyonu
        out = self.wo(out)

        # Spike firing rate takibi (Sadece V darbe ürettiği için)
        spk_v = (v > 0).float().mean()
        self.last_spk_rate = spk_v.detach()

        new_mem = {"v": mem_v}
        return out, new_mem

    def init_mems(self, batch: int, device: torch.device) -> Dict[str, torch.Tensor]:
        return {"v": self.lif_v.init_leaky()}


class SpikingFFN(nn.Module):
    """Spiking Feed-Forward Network (SwiGLU gating + LIF nöronlar)."""

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
        h = a * b  # Spiking gating (ikili çarpım / AND benzeri aktivasyon)
        out, m3 = self.lif3(self.w3(h), mems["3"])

        spk_a = (a > 0).float().mean()
        spk_b = (b > 0).float().mean()
        spk_out = (out > 0).float().mean()
        self.last_spk_rate = ((spk_a + spk_b + spk_out) / 3.0).detach()

        return out, {"1": m1, "2": m2, "3": m3}

    def init_mems(self, batch: int, device: torch.device) -> Dict[str, torch.Tensor]:
        return {n: getattr(self, f"lif{n}").init_leaky() for n in ["1", "2", "3"]}


class HybridTransformerBlock(nn.Module):
    """Pre-LayerNorm ve Residual bağlantılı Hibrit Transformer Bloğu."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_hidden: Optional[int] = None,
        beta: float = 0.9,
        threshold: float = 0.5,
    ):
        super().__init__()
        ffn_hidden = ffn_hidden or dim * 2
        self.norm1 = nn.LayerNorm(dim)
        self.attn = HybridSelfAttention(dim, num_heads, beta, threshold)
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


class HybridLIFTransformerLM(nn.Module):
    """Hibrit LIF + Attention Dil Modeli (Stage 8).

    Float Q, K doğrusal projeksiyonları ve Softmax causal attention ile darbe (spike) V birleşimi.
    FFN katmanında LIF nöronları ile darbe dinamiklerini sürdürür.
    """

    def __init__(
        self,
        vocab_size: int = 16384,
        embed_dim: int = 512,
        num_blocks: int = 8,
        num_heads: int = 8,
        ffn_hidden: int = 1024,
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
        self.is_hybrid_lif_attn = True

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(max_seq, embed_dim)  # Pozisyonel gömme
        self.blocks = nn.ModuleList([
            HybridTransformerBlock(embed_dim, num_heads, ffn_hidden, beta, threshold)
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

        # Ortalama spike oranını hesapla (Attention V + FFN darbe oranları)
        rates = [blk.attn.last_spk_rate for blk in self.blocks] + [blk.ffn.last_spk_rate for blk in self.blocks]
        spk_rate = torch.stack(rates).mean().detach()
        self.last_spk_rate = spk_rate

        if return_spikes:
            return logits, spk_rate
        return logits, spk_rate

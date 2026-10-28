"""Mergen-SNN — Stage 3: BPE Tabanlı SNN Dil Modeli (Mergen-Pico, ~14M Parametre)."""

import snntorch as snn
import torch
import torch.nn as nn

try:
    from mergen.cython_ext.fast_lif import (
        fast_lif_forward,
        fast_lif_sequence_backward,
        fast_lif_sequence_forward,
    )
except ImportError:
    fast_lif_forward = None
    fast_lif_sequence_forward = None
    fast_lif_sequence_backward = None


class FastLIFFunction(torch.autograd.Function):
    """Tek zaman adımlı Cython LIF ileri geçişi ve PyTorch ATan surrogate türev fonksiyonu."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        mem: torch.Tensor,
        beta: float = 0.9,
        threshold: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if fast_lif_forward is None:
            raise RuntimeError(
                "Cython fast_lif modülü derlenmemiş! 'uv run python src/mergen/cython_ext/setup.py build_ext --inplace' çalıştırın."
            )

        x_c = x.contiguous()
        mem_c = mem.contiguous()
        batch_size, hidden_size = x_c.shape

        spk = torch.empty((batch_size, hidden_size), dtype=torch.float32)
        new_mem = torch.empty((batch_size, hidden_size), dtype=torch.float32)

        fast_lif_forward(
            x_c.numpy(),
            mem_c.numpy(),
            float(beta),
            float(threshold),
            spk_out=spk.numpy(),
            mem_out=new_mem.numpy(),
        )

        m = mem_c * beta + x_c
        ctx.save_for_backward(m)
        ctx.beta = beta
        ctx.threshold = threshold

        return spk, new_mem

    @staticmethod
    def backward(ctx, grad_spk: torch.Tensor, grad_mem: torch.Tensor):
        (m,) = ctx.saved_tensors
        beta = ctx.beta
        threshold = ctx.threshold

        grad_m = None

        if grad_spk is not None:
            pi = 3.141592653589793
            shift = m - threshold
            surr = 1.0 / (1.0 + (pi * shift) ** 2)
            grad_m = grad_spk * surr

        if grad_mem is not None:
            grad_m = (grad_m + grad_mem) if grad_m is not None else grad_mem.clone()

        if grad_m is None:
            return None, None, None, None

        grad_x = grad_m
        grad_mem_in = grad_m * beta

        return grad_x, grad_mem_in, None, None


class FastLIFSequence(torch.autograd.Function):
    """Tüm zaman adımlarını (num_steps) Cython içinde tek seferde işleten autograd sarmalayıcısı."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        mem_init: torch.Tensor,
        beta: float = 0.9,
        threshold: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if fast_lif_sequence_forward is None:
            raise RuntimeError(
                "Cython fast_lif_sequence_forward bulunamadı! Lütfen extension'ı derleyin."
            )

        x_c = x.contiguous()
        batch_size = x_c.size(0)
        hidden_size = x_c.size(2)

        if mem_init is None or mem_init.numel() == 0 or mem_init.shape != (batch_size, hidden_size):
            mem_init_c = torch.zeros((batch_size, hidden_size), dtype=torch.float32, device=x.device)
        else:
            mem_init_c = mem_init.contiguous()

        spk_np, mem_last_np, mem_all_np = fast_lif_sequence_forward(
            x_c.numpy(), mem_init_c.numpy(), float(beta), float(threshold), True
        )

        spk_out = torch.from_numpy(spk_np)
        mem_last_out = torch.from_numpy(mem_last_np)
        mem_all = torch.from_numpy(mem_all_np)

        ctx.save_for_backward(spk_out, mem_all, mem_init_c)
        ctx.beta = float(beta)
        ctx.threshold = float(threshold)

        return spk_out, mem_last_out

    @staticmethod
    def backward(ctx, grad_spk_out: torch.Tensor, grad_mem_out: torch.Tensor):
        spk_out, mem_all, mem_init = ctx.saved_tensors
        beta = ctx.beta
        threshold = ctx.threshold

        grad_spk_c = grad_spk_out.contiguous() if grad_spk_out is not None else torch.zeros_like(spk_out)
        grad_mem_np = grad_mem_out.contiguous().numpy() if grad_mem_out is not None else None

        grad_x_np, grad_mem_init_np = fast_lif_sequence_backward(
            grad_spk_c.numpy(),
            spk_out.numpy(),
            mem_all.numpy(),
            mem_init.numpy(),
            beta,
            threshold,
            2.0,
            grad_mem_np,
        )

        return torch.from_numpy(grad_x_np), torch.from_numpy(grad_mem_init_np), None, None


class FastLeaky(nn.Module):
    """Cython ile hızlandırılmış Leaky Integrate-and-Fire (LIF) nöron katmanı.

    snnTorch snn.Leaky sınıfı için doğrudan (drop-in) ikamedir.
    """

    def __init__(self, beta: float = 0.9, threshold: float = 1.0):
        super().__init__()
        self.beta = float(beta)
        self.threshold = float(threshold)

    def init_leaky(self) -> torch.Tensor:
        return torch.empty(0)

    def forward(
        self, input_: torch.Tensor, mem: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_.dim() == 3:
            # Dizi girişi: (batch, num_steps, hidden)
            return FastLIFSequence.apply(input_, mem, self.beta, self.threshold)
        if mem is None or mem.numel() == 0 or mem.shape != input_.shape:
            mem = torch.zeros_like(input_)
        return FastLIFFunction.apply(input_, mem, self.beta, self.threshold)


class BpeSNNLM(nn.Module):
    """BPE tabanlı Spiking Sinir Ağı Dil Modeli (Mergen-Pico).

    Her timestep bir BPE alt-kelimesi (token) işler.
    LIF membran potansiyelleri zaman boyunca taşınarak bağlam korunur.
    Embedding çıkışı FC1 ve FC2 katmanları ile LIF nöronları üzerinden iletilir.
    Cython Sequence LIF ile döngüsüz yüksek hızlı hesaplama sağlar.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        embed_dim: int = 512,
        hidden_size: int = 1024,
        beta: float = 0.9,
        use_cython: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.beta = beta
        self.use_cython = use_cython and (fast_lif_sequence_forward is not None)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_size)

        if self.use_cython:
            self.lif1 = FastLeaky(beta=beta)
            self.lif2 = FastLeaky(beta=beta)
        else:
            self.lif1 = snn.Leaky(beta=beta)
            self.lif2 = snn.Leaky(beta=beta)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, num_steps: int = 64) -> torch.Tensor:
        emb = self.embedding(input_ids)  # (batch, num_steps, embed_dim)
        cur1_all = self.fc1(emb)         # (batch, num_steps, hidden)

        if self.use_cython:
            # Tek bir Cython çağrısı ile tüm timestep'ler (Python döngüsü yok!)
            spk1_all, _ = FastLIFSequence.apply(
                cur1_all, self.lif1.init_leaky(), self.beta, 1.0
            )

            cur2_all = self.fc2(spk1_all)    # (batch, num_steps, hidden)
            spk2_all, _ = FastLIFSequence.apply(
                cur2_all, self.lif2.init_leaky(), self.beta, 1.0
            )
        else:
            # snnTorch fallback döngüsü
            mem1 = self.lif1.init_leaky()
            spk1_list = []
            for t in range(num_steps):
                spk1, mem1 = self.lif1(cur1_all[:, t, :].contiguous(), mem1)
                spk1_list.append(spk1)
            spk1_all = torch.stack(spk1_list, dim=1)

            cur2_all = self.fc2(spk1_all)
            mem2 = self.lif2.init_leaky()
            spk2_list = []
            for t in range(num_steps):
                spk2, mem2 = self.lif2(cur2_all[:, t, :].contiguous(), mem2)
                spk2_list.append(spk2)
            spk2_all = torch.stack(spk2_list, dim=1)

        logits_all = self.fc_out(spk2_all)  # (batch, num_steps, vocab)
        return logits_all

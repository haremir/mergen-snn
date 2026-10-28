"""Mergen-SNN — Stage 3: BPE Tabanlı SNN Dil Modeli (Mergen-Pico, ~14M Parametre)."""

import snntorch as snn
import torch
import torch.nn as nn

try:
    from mergen.cython_ext.fast_lif import fast_lif_forward
except ImportError:
    fast_lif_forward = None


class FastLIFFunction(torch.autograd.Function):
    """Cython hızlandırmalı LIF ileri geçişi ve PyTorch ATan surrogate türev fonksiyonu."""

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

        # Tensörleri doğrudan PyTorch belleğinde ayırıp numpy view üzerinden C++'a veriyoruz (sıfır kopyalama)
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

        # Geriye doğru geçişte surrogate gradient türevi için membran potansiyeli m kaydedilir
        # m = beta * mem + x
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
            # snnTorch ile birebir uyumlu ATan vekil gradyanı (alpha = 2.0)
            # d(spk)/d(m) = 1.0 / (1.0 + (pi * (m - threshold))^2)
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
        if mem is None or mem.numel() == 0 or mem.shape != input_.shape:
            mem = torch.zeros_like(input_)
        return FastLIFFunction.apply(input_, mem, self.beta, self.threshold)


class BpeSNNLM(nn.Module):
    """BPE tabanlı Spiking Sinir Ağı Dil Modeli (Mergen-Pico).

    Her timestep bir BPE alt-kelimesi (token) işler.
    LIF membran potansiyelleri zaman boyunca taşınarak bağlam korunur.
    Embedding çıkışı FC1 ve FC2 katmanları ile LIF nöronları üzerinden iletilir.
    Cython hızlandırmalı FastLeaky veya snnTorch Leaky ile çalışabilir.
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
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_size)

        if use_cython and fast_lif_forward is not None:
            self.lif1 = FastLeaky(beta=beta)
            self.lif2 = FastLeaky(beta=beta)
        else:
            self.lif1 = snn.Leaky(beta=beta)
            self.lif2 = snn.Leaky(beta=beta)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, num_steps: int = 128) -> torch.Tensor:
        batch_size = input_ids.size(0)
        emb = self.embedding(input_ids)  # (batch, num_steps, embed_dim)

        # 1. FC1'i TÜM timestep'ler için TEK SEFERDE hesapla (PyTorch BLAS kullanır)
        cur1_all = self.fc1(emb)  # (batch, num_steps, hidden_size)

        # 2. LIF1 döngüsü (Cython FastLIF)
        mem1 = self.lif1.init_leaky()
        spk1_list = []
        for t in range(num_steps):
            # .contiguous() ekle, Cython buffer uyumu için
            spk1, mem1 = self.lif1(cur1_all[:, t, :].contiguous(), mem1)
            spk1_list.append(spk1)
        spk1_all = torch.stack(spk1_list, dim=1)  # (batch, num_steps, hidden_size)

        # 3. FC2'yi TEK SEFERDE hesapla
        cur2_all = self.fc2(spk1_all)  # (batch, num_steps, hidden_size)

        # 4. LIF2 döngüsü (Cython FastLIF)
        mem2 = self.lif2.init_leaky()
        spk2_list = []
        for t in range(num_steps):
            spk2, mem2 = self.lif2(cur2_all[:, t, :].contiguous(), mem2)
            spk2_list.append(spk2)
        spk2_all = torch.stack(spk2_list, dim=1)

        # 5. FC_OUT'u TEK SEFERDE hesapla
        logits_all = self.fc_out(spk2_all)  # (batch, num_steps, vocab_size)

        return logits_all

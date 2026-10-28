"""Mergen-SNN — Stage 3: BPE Tabanlı SNN Dil Modeli (Mergen-Pico, ~14M Parametre)."""

import math
import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

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

try:
    from mergen.cython_ext.fast_gemm import fast_gemm_forward
except ImportError:
    fast_gemm_forward = None


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
            return FastLIFSequence.apply(input_, mem, self.beta, self.threshold)
        if mem is None or mem.numel() == 0 or mem.shape != input_.shape:
            mem = torch.zeros_like(input_)
        return FastLIFFunction.apply(input_, mem, self.beta, self.threshold)


class FastLinearFunction(torch.autograd.Function):
    """Cython AVX2 SIMD GEMM ileri ve geri yayılım autograd fonksiyonu."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
        ctx.save_for_backward(x, weight, bias)
        x_c = x.contiguous().float()
        orig_shape = x_c.shape
        in_features = orig_shape[-1]
        out_features = weight.size(0)

        x_2d = x_c.view(-1, in_features)
        w_2d = weight.contiguous().float()
        b_1d = bias.contiguous().float() if bias is not None else None

        out_np = fast_gemm_forward(x_2d.numpy(), w_2d.numpy(), b_1d, transposed_b=True)
        out_t = torch.from_numpy(out_np)

        if len(orig_shape) > 2:
            out_t = out_t.view(*orig_shape[:-1], out_features)

        return out_t

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, bias = ctx.saved_tensors
        grad_output_c = grad_output.contiguous().float()
        orig_shape = x.shape
        in_features = orig_shape[-1]
        out_features = weight.size(0)

        grad_2d = grad_output_c.view(-1, out_features)
        x_2d = x.contiguous().float().view(-1, in_features)
        w_2d = weight.contiguous().float()

        # grad_x = grad_output @ weight (transposed_b=False)
        grad_x_np = fast_gemm_forward(grad_2d.numpy(), w_2d.numpy(), transposed_b=False)
        grad_x = torch.from_numpy(grad_x_np).view(*orig_shape)

        # grad_weight = grad_output.T @ x (transposed_b=False)
        grad_w_np = fast_gemm_forward(grad_2d.t().numpy(), x_2d.numpy(), transposed_b=False)
        grad_weight = torch.from_numpy(grad_w_np)

        grad_bias = None
        if bias is not None:
            grad_bias = grad_2d.sum(dim=0)

        return grad_x, grad_weight, grad_bias


class FastLinear(nn.Module):
    """Cython AVX2 SIMD GEMM hızlandırmalı Lineer katman."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if fast_gemm_forward is not None:
            return FastLinearFunction.apply(x, self.weight, self.bias)
        return F.linear(x, self.weight, self.bias)


class BpeSNNLM(nn.Module):
    """BPE tabanlı Spiking Sinir Ağı Dil Modeli (Mergen-Pico).

    Her timestep bir BPE alt-kelimesi (token) işler.
    LIF membran potansiyelleri zaman boyunca taşınarak bağlam korunur.
    Embedding çıkışı FC1 ve FC2 katmanları ile LIF nöronları üzerinden iletilir.
    Cython Sequence LIF ve isteğe bağlı Cython FastLinear ile çalışabilir.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        embed_dim: int = 512,
        hidden_size: int = 1024,
        beta: float = 0.9,
        use_cython: bool = True,
        use_fast_linear: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.beta = beta
        self.use_cython = use_cython and (fast_lif_sequence_forward is not None)
        self.use_fast_linear = use_fast_linear and (fast_gemm_forward is not None)

        linear_cls = FastLinear if self.use_fast_linear else nn.Linear

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = linear_cls(embed_dim, hidden_size)

        if self.use_cython:
            self.lif1 = FastLeaky(beta=beta)
            self.lif2 = FastLeaky(beta=beta)
        else:
            self.lif1 = snn.Leaky(beta=beta)
            self.lif2 = snn.Leaky(beta=beta)

        self.fc2 = linear_cls(hidden_size, hidden_size)
        self.fc_out = linear_cls(hidden_size, vocab_size)

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

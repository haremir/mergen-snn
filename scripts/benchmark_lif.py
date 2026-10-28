"""Mergen-SNN — Stage 3: PyTorch Linear vs Cython FastLinear (AVX2 GEMM) Benchmark.

10 eğitim adımı (batch=64, seq_len=64, vocab=8192) üzerinden:
- PyTorch Linear (MKL/oneDNN BLAS) + Cython FastLIFSequence
- Cython FastLinear (AVX2 SIMD GEMM) + Cython FastLIFSequence
karşılaştırılır ve adım süreleri raporlanır.
"""

import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

import sys
import time
import torch
import torch.nn.functional as F

# Windows konsol UTF-8 çıktı desteği
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from mergen.models.snn_lm_bpe import BpeSNNLM


def run_benchmark(steps: int = 10, batch_size: int = 64, seq_len: int = 64):
    print("=" * 70)
    print(" MERGEN-SNN GEMM BENCHMARK: PyTorch Linear vs Cython FastLinear (AVX2)")
    print(f" Ayarlar: Batch={batch_size}, Seq={seq_len}, Timesteps={seq_len}, Steps={steps}")
    print(f" Thread Ayarı: OMP={os.environ.get('OMP_NUM_THREADS')}, MKL={os.environ.get('MKL_NUM_THREADS')}")
    print("=" * 70)

    torch.manual_seed(42)
    device = torch.device("cpu")

    print("\nModeller yükleniyor (14M BpeSNNLM)...")
    # Model 1: PyTorch BLAS Linear + Cython FastLIFSequence
    model_pytorch = BpeSNNLM(
        vocab_size=8192,
        embed_dim=512,
        hidden_size=1024,
        beta=0.9,
        use_cython=True,
        use_fast_linear=False,
    ).to(device)

    # Model 2: Cython FastLinear (AVX2 GEMM) + Cython FastLIFSequence
    model_fast_linear = BpeSNNLM(
        vocab_size=8192,
        embed_dim=512,
        hidden_size=1024,
        beta=0.9,
        use_cython=True,
        use_fast_linear=True,
    ).to(device)

    opt_pytorch = torch.optim.SGD(model_pytorch.parameters(), lr=0.003, momentum=0.9)
    opt_fast = torch.optim.SGD(model_fast_linear.parameters(), lr=0.003, momentum=0.9)

    # Sentetik veri
    x = torch.randint(0, 8192, (batch_size, seq_len), device=device)
    y = torch.randint(0, 8192, (batch_size, seq_len), device=device)

    # 1. Warm-up
    print("Isınma adımları (warm-up) çalıştırılıyor...")
    out = model_pytorch(x, num_steps=seq_len)
    loss = F.cross_entropy(out.reshape(-1, 8192), y.reshape(-1))
    loss.backward()
    opt_pytorch.step()
    opt_pytorch.zero_grad()

    out = model_fast_linear(x, num_steps=seq_len)
    loss = F.cross_entropy(out.reshape(-1, 8192), y.reshape(-1))
    loss.backward()
    opt_fast.step()
    opt_fast.zero_grad()

    # 2. PyTorch Linear Ölçümü
    print("\n[1/2] PyTorch Linear Modeli Test Ediliyor (10 adım)...")
    pytorch_times = []
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        opt_pytorch.zero_grad()
        out = model_pytorch(x, num_steps=seq_len)
        loss = F.cross_entropy(out.reshape(-1, 8192), y.reshape(-1))
        loss.backward()
        opt_pytorch.step()
        dt = time.perf_counter() - t0
        pytorch_times.append(dt)
        print(f"  Step {step:2d}/10: {dt:.4f} s")

    # 3. Cython FastLinear Ölçümü
    print("\n[2/2] Cython FastLinear (AVX2 GEMM) Modeli Test Ediliyor (10 adım)...")
    fast_times = []
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        opt_fast.zero_grad()
        out = model_fast_linear(x, num_steps=seq_len)
        loss = F.cross_entropy(out.reshape(-1, 8192), y.reshape(-1))
        loss.backward()
        opt_fast.step()
        dt = time.perf_counter() - t0
        fast_times.append(dt)
        print(f"  Step {step:2d}/10: {dt:.4f} s")

    # Sonuçlar
    avg_pytorch = sum(pytorch_times) / len(pytorch_times)
    avg_fast = sum(fast_times) / len(fast_times)
    speedup = avg_pytorch / avg_fast

    print("\n" + "=" * 70)
    print(" BENCHMARK SONUÇLARI")
    print("=" * 70)
    print(f" PyTorch Linear Süresi (ort. 1 step)       : {avg_pytorch:.4f} s")
    print(f" Cython FastLinear Süresi (ort. 1 step)    : {avg_fast:.4f} s")
    print(f" Hızlanma / Karşılaştırma Oranı (Speedup) : {speedup:.2f}x")
    print("=" * 70)

    return avg_pytorch, avg_fast, speedup


if __name__ == "__main__":
    run_benchmark(steps=10, batch_size=64, seq_len=64)

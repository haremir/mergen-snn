"""Mergen-SNN — Stage 3: Cython vs snnTorch LIF Hız Karşılaştırması (Benchmark).

10 eğitim adımı (batch=32, seq_len=128, vocab=8192) üzerinden:
- Eski snnTorch Leaky modeli
- Yeni Cython FastLeaky modeli
karşılaştırılır ve hızlanma oranı hesaplanır.
"""

import sys
import time
import torch
import torch.nn.functional as F

# Windows konsol UTF-8 çıktı desteği
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from mergen.models.snn_lm_bpe import BpeSNNLM


def run_benchmark(steps: int = 10, batch_size: int = 32, seq_len: int = 128):
    print("=" * 65)
    print(" MERGEN-SNN LIF BENCHMARK: snnTorch vs Cython FastLeaky")
    print(f" Ayarlar: Batch={batch_size}, Seq={seq_len}, Timesteps={seq_len}, Steps={steps}")
    print("=" * 65)

    torch.manual_seed(42)
    device = torch.device("cpu")

    # Modelleri oluştur (~14M parametre)
    print("\nModeller yükleniyor (14M BpeSNNLM)...")
    model_snntorch = BpeSNNLM(
        vocab_size=8192,
        embed_dim=512,
        hidden_size=1024,
        beta=0.9,
        use_cython=False,
    ).to(device)

    model_cython = BpeSNNLM(
        vocab_size=8192,
        embed_dim=512,
        hidden_size=1024,
        beta=0.9,
        use_cython=True,
    ).to(device)

    opt_snntorch = torch.optim.AdamW(model_snntorch.parameters(), lr=1e-3)
    opt_cython = torch.optim.AdamW(model_cython.parameters(), lr=1e-3)

    # Sentetik veri
    x = torch.randint(0, 8192, (batch_size, seq_len), device=device)
    y = torch.randint(0, 8192, (batch_size, seq_len), device=device)

    # 1. Warm-up
    print("Isınma adımları (warm-up) çalıştırılıyor...")
    out = model_snntorch(x, num_steps=seq_len)
    loss = F.cross_entropy(out.reshape(-1, 8192), y.reshape(-1))
    loss.backward()
    opt_snntorch.step()
    opt_snntorch.zero_grad()

    out = model_cython(x, num_steps=seq_len)
    loss = F.cross_entropy(out.reshape(-1, 8192), y.reshape(-1))
    loss.backward()
    opt_cython.step()
    opt_cython.zero_grad()

    # 2. snnTorch Ölçümü
    print("\n[1/2] snnTorch Modeli Test Ediliyor (10 adım)...")
    snntorch_times = []
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        opt_snntorch.zero_grad()
        out = model_snntorch(x, num_steps=seq_len)
        loss = F.cross_entropy(out.reshape(-1, 8192), y.reshape(-1))
        loss.backward()
        opt_snntorch.step()
        dt = time.perf_counter() - t0
        snntorch_times.append(dt)
        print(f"  Step {step:2d}/10: {dt:.4f} s")

    # 3. Cython Ölçümü
    print("\n[2/2] Cython FastLeaky Modeli Test Ediliyor (10 adım)...")
    cython_times = []
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        opt_cython.zero_grad()
        out = model_cython(x, num_steps=seq_len)
        loss = F.cross_entropy(out.reshape(-1, 8192), y.reshape(-1))
        loss.backward()
        opt_cython.step()
        dt = time.perf_counter() - t0
        cython_times.append(dt)
        print(f"  Step {step:2d}/10: {dt:.4f} s")

    # Sonuçlar
    avg_snntorch = sum(snntorch_times) / len(snntorch_times)
    avg_cython = sum(cython_times) / len(cython_times)
    speedup = avg_snntorch / avg_cython

    print("\n" + "=" * 65)
    print(" BENCHMARK SONUÇLARI")
    print("=" * 65)
    print(f" Eski snnTorch Süresi (ortalama 1 step): {avg_snntorch:.4f} s")
    print(f" Yeni Cython Süresi   (ortalama 1 step): {avg_cython:.4f} s")
    print(f" Hızlanma Oranı (Speedup)              : {speedup:.2f}x ({speedup:.2f} kat hızlandı)")
    print("=" * 65)

    return avg_snntorch, avg_cython, speedup


if __name__ == "__main__":
    run_benchmark(steps=10, batch_size=32, seq_len=128)

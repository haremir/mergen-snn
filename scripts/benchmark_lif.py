"""Mergen-SNN — Stage 3: Eski (Python Döngülü) vs Yeni (Cython Sequence LIF) Benchmark.

10 eğitim adımı (batch=64, seq_len=64, vocab=8192) üzerinden:
- Eski model: Python for döngüsü ile işletilen LIF modeli
- Yeni model: Tek Cython çağrısı ile işletilen FastLIFSequence modeli
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


def run_benchmark(steps: int = 10, batch_size: int = 64, seq_len: int = 64):
    print("=" * 70)
    print(" MERGEN-SNN BENCHMARK: Eski (Python Döngülü) vs Yeni (Cython Sequence LIF)")
    print(f" Ayarlar: Batch={batch_size}, Seq={seq_len}, Timesteps={seq_len}, Steps={steps}")
    print("=" * 70)

    torch.manual_seed(42)
    device = torch.device("cpu")

    print("\nModeller yükleniyor (14M BpeSNNLM)...")
    model_old = BpeSNNLM(
        vocab_size=8192,
        embed_dim=512,
        hidden_size=1024,
        beta=0.9,
        use_cython=False,
    ).to(device)

    model_new = BpeSNNLM(
        vocab_size=8192,
        embed_dim=512,
        hidden_size=1024,
        beta=0.9,
        use_cython=True,
    ).to(device)

    opt_old = torch.optim.SGD(model_old.parameters(), lr=0.003, momentum=0.9)
    opt_new = torch.optim.SGD(model_new.parameters(), lr=0.003, momentum=0.9)

    # Sentetik veri
    x = torch.randint(0, 8192, (batch_size, seq_len), device=device)
    y = torch.randint(0, 8192, (batch_size, seq_len), device=device)

    # 1. Warm-up
    print("Isınma adımları (warm-up) çalıştırılıyor...")
    out = model_old(x, num_steps=seq_len)
    loss = F.cross_entropy(out.reshape(-1, 8192), y.reshape(-1))
    loss.backward()
    opt_old.step()
    opt_old.zero_grad()

    out = model_new(x, num_steps=seq_len)
    loss = F.cross_entropy(out.reshape(-1, 8192), y.reshape(-1))
    loss.backward()
    opt_new.step()
    opt_new.zero_grad()

    # 2. Eski Model Ölçümü
    print("\n[1/2] Eski Model (Python Döngülü) Test Ediliyor (10 adım)...")
    old_times = []
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        opt_old.zero_grad()
        out = model_old(x, num_steps=seq_len)
        loss = F.cross_entropy(out.reshape(-1, 8192), y.reshape(-1))
        loss.backward()
        opt_old.step()
        dt = time.perf_counter() - t0
        old_times.append(dt)
        print(f"  Step {step:2d}/10: {dt:.4f} s")

    # 3. Yeni Vektörize Model Ölçümü
    print("\n[2/2] Yeni Model (Cython Sequence LIF) Test Ediliyor (10 adım)...")
    new_times = []
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        opt_new.zero_grad()
        out = model_new(x, num_steps=seq_len)
        loss = F.cross_entropy(out.reshape(-1, 8192), y.reshape(-1))
        loss.backward()
        opt_new.step()
        dt = time.perf_counter() - t0
        new_times.append(dt)
        print(f"  Step {step:2d}/10: {dt:.4f} s")

    # Sonuçlar
    avg_old = sum(old_times) / len(old_times)
    avg_new = sum(new_times) / len(new_times)
    speedup = avg_old / avg_new

    print("\n" + "=" * 70)
    print(" BENCHMARK SONUÇLARI")
    print("=" * 70)
    print(f" Eski Model Süresi (Python Döngülü, ort. 1 step): {avg_old:.4f} s")
    print(f" Yeni Model Süresi (Cython Sequence, ort. 1 step): {avg_new:.4f} s")
    print(f" Hızlanma Oranı (Speedup)                         : {speedup:.2f}x ({speedup:.2f} kat hızlandı)")
    print("=" * 70)

    return avg_old, avg_new, speedup


if __name__ == "__main__":
    run_benchmark(steps=10, batch_size=64, seq_len=64)

"""Mergen-SNN — Stage 3: Eski (Döngü içi Lineer) vs Yeni (Vektörize Lineer + Cython) Benchmark.

10 eğitim adımı (batch=64, seq_len=64, vocab=8192) üzerinden:
- Eski model: Döngü içi Lineer katmanlar (fc1, fc2, fc_out) ve snnTorch
- Yeni model: Sequence Batching (Vektörize Lineer) ve Cython FastLeaky
karşılaştırılır ve hızlanma oranı hesaplanır.
"""

import sys
import time
import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

# Windows konsol UTF-8 çıktı desteği
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from mergen.models.snn_lm_bpe import BpeSNNLM


class OldLoopBpeSNNLM(nn.Module):
    """Eski Model: Her zaman adımında döngü içinde çağrılan Lineer katmanlar."""

    def __init__(
        self,
        vocab_size: int = 8192,
        embed_dim: int = 512,
        hidden_size: int = 1024,
        beta: float = 0.9,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, num_steps: int = 64) -> torch.Tensor:
        emb = self.embedding(input_ids)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        logits_list = []
        for t in range(num_steps):
            x_t = emb[:, t, :]
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            logits_t = self.fc_out(spk2)
            logits_list.append(logits_t)
        return torch.stack(logits_list, dim=1)


def run_benchmark(steps: int = 10, batch_size: int = 64, seq_len: int = 64):
    print("=" * 70)
    print(" MERGEN-SNN BENCHMARK: Eski (Döngü içi Lineer) vs Yeni (Vektörize Lineer)")
    print(f" Ayarlar: Batch={batch_size}, Seq={seq_len}, Timesteps={seq_len}, Steps={steps}")
    print("=" * 70)

    torch.manual_seed(42)
    device = torch.device("cpu")

    print("\nModeller yükleniyor (14M parametre)...")
    model_old = OldLoopBpeSNNLM(
        vocab_size=8192,
        embed_dim=512,
        hidden_size=1024,
        beta=0.9,
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
    print("\n[1/2] Eski Model (Döngü içi Lineer) Test Ediliyor (10 adım)...")
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
    print("\n[2/2] Yeni Model (Vektörize Lineer + Cython) Test Ediliyor (10 adım)...")
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
    print(f" Eski Model Süresi (döngü içi Lineer, ort. 1 step): {avg_old:.4f} s")
    print(f" Yeni Model Süresi (vektörize Lineer, ort. 1 step): {avg_new:.4f} s")
    print(f" Hızlanma Oranı (Speedup)                          : {speedup:.2f}x ({speedup:.2f} kat hızlandı)")
    print("=" * 70)

    return avg_old, avg_new, speedup


if __name__ == "__main__":
    run_benchmark(steps=10, batch_size=64, seq_len=64)

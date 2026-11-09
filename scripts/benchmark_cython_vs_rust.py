"""Mergen-SNN — Cython vs Rust LIF Kernel Benchmark (Aşama 6, 50M Konfigürasyonu)."""

import time
import numpy as np

try:
    import mergen_rust
except ImportError:
    mergen_rust = None

try:
    import mergen.cython_ext.fast_lif as cython_lif
except ImportError:
    cython_lif = None


def run_benchmark(batch_size: int = 16, num_steps: int = 32, hidden_size: int = 1400, iters: int = 100):
    print("=" * 70)
    print("Mergen-SNN — Cython vs Rust LIF Kernel Benchmark")
    print("=" * 70)
    print(f"Konfigürasyon: Batch={batch_size}, NumSteps={num_steps}, Hidden={hidden_size}")
    print(f"Toplam Eleman / Adım: {batch_size * num_steps * hidden_size:,} | İterasyon: {iters}")
    print("=" * 70)

    if cython_lif is None:
        raise RuntimeError("Cython fast_lif modülü yüklenemedi!")
    if mergen_rust is None:
        raise RuntimeError("mergen_rust modülü yüklenemedi!")

    np.random.seed(42)
    x = np.random.randn(batch_size, num_steps, hidden_size).astype(np.float32)
    mem_init = np.zeros((batch_size, hidden_size), dtype=np.float32)
    grad_spk = np.random.randn(batch_size, num_steps, hidden_size).astype(np.float32)

    beta = 0.97
    threshold = 0.5
    alpha = 2.0

    # 1. Doğruluk Kontrolü (Sanity check)
    spk_cy, mout_cy, mall_cy = cython_lif.fast_lif_sequence_forward(x, mem_init, beta, threshold, True)
    spk_rs, mout_rs, mall_rs = mergen_rust.fast_lif_sequence_forward(x, mem_init, beta, threshold, True)

    assert np.array_equal(spk_cy, spk_rs), "İleri geçiş spike uyuşmazlığı!"
    assert np.allclose(mout_cy, mout_rs, atol=1e-5), "İleri geçiş son membran uyuşmazlığı!"
    assert np.allclose(mall_cy, mall_rs, atol=1e-5), "İleri geçiş tüm membran uyuşmazlığı!"

    gx_cy, gm_cy = cython_lif.fast_lif_sequence_backward(grad_spk, spk_cy, mall_cy, mem_init, beta, threshold, alpha)
    gx_rs, gm_rs = mergen_rust.fast_lif_sequence_backward(grad_spk, spk_rs, mall_rs, mem_init, beta, threshold, alpha)

    assert np.allclose(gx_cy, gx_rs, atol=1e-5), "Geriye yayılım grad_x uyuşmazlığı!"
    assert np.allclose(gm_cy, gm_rs, atol=1e-5), "Geriye yayılım grad_mem_init uyuşmazlığı!"
    print("[OK] Doğrulama: Cython ve Rust çıktıları 100% matematiksel olarak eşleşiyor.\n")

    # Warmup
    for _ in range(10):
        cython_lif.fast_lif_sequence_forward(x, mem_init, beta, threshold, True)
        cython_lif.fast_lif_sequence_backward(grad_spk, spk_cy, mall_cy, mem_init, beta, threshold, alpha)
        mergen_rust.fast_lif_sequence_forward(x, mem_init, beta, threshold, True)
        mergen_rust.fast_lif_sequence_backward(grad_spk, spk_rs, mall_rs, mem_init, beta, threshold, alpha)

    # 2. Benchmark - Cython Forward
    cy_fwd_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        cython_lif.fast_lif_sequence_forward(x, mem_init, beta, threshold, True)
        cy_fwd_times.append((time.perf_counter() - t0) * 1000.0)

    # 3. Benchmark - Rust Forward
    rs_fwd_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mergen_rust.fast_lif_sequence_forward(x, mem_init, beta, threshold, True)
        rs_fwd_times.append((time.perf_counter() - t0) * 1000.0)

    # 4. Benchmark - Cython Backward
    cy_bwd_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        cython_lif.fast_lif_sequence_backward(grad_spk, spk_cy, mall_cy, mem_init, beta, threshold, alpha)
        cy_bwd_times.append((time.perf_counter() - t0) * 1000.0)

    # 5. Benchmark - Rust Backward
    rs_bwd_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mergen_rust.fast_lif_sequence_backward(grad_spk, spk_rs, mall_rs, mem_init, beta, threshold, alpha)
        rs_bwd_times.append((time.perf_counter() - t0) * 1000.0)

    cy_fwd_avg = float(np.mean(cy_fwd_times))
    rs_fwd_avg = float(np.mean(rs_fwd_times))
    cy_bwd_avg = float(np.mean(cy_bwd_times))
    rs_bwd_avg = float(np.mean(rs_bwd_times))

    cy_tot_avg = cy_fwd_avg + cy_bwd_avg
    rs_tot_avg = rs_fwd_avg + rs_bwd_avg

    speedup_fwd = cy_fwd_avg / rs_fwd_avg if rs_fwd_avg > 0 else 0
    speedup_bwd = cy_bwd_avg / rs_bwd_avg if rs_bwd_avg > 0 else 0
    speedup_tot = cy_tot_avg / rs_tot_avg if rs_tot_avg > 0 else 0

    print("-" * 70)
    print(f"{'İşlem':<20} | {'Cython (ms)':<15} | {'Rust (ms)':<15} | {'Hızlanma':<12}")
    print("-" * 70)
    print(f"{'Forward':<20} | {cy_fwd_avg:>12.3f} ms | {rs_fwd_avg:>12.3f} ms | {speedup_fwd:>10.2f}x")
    print(f"{'Backward':<20} | {cy_bwd_avg:>12.3f} ms | {rs_bwd_avg:>12.3f} ms | {speedup_bwd:>10.2f}x")
    print("-" * 70)
    print(f"{'Toplam (Fwd + Bwd)':<20} | {cy_tot_avg:>12.3f} ms | {rs_tot_avg:>12.3f} ms | {speedup_tot:>10.2f}x")
    print("=" * 70)

    return {
        "cy_fwd": cy_fwd_avg,
        "rs_fwd": rs_fwd_avg,
        "cy_bwd": cy_bwd_avg,
        "rs_bwd": rs_bwd_avg,
        "cy_tot": cy_tot_avg,
        "rs_tot": rs_tot_avg,
        "speedup_tot": speedup_tot,
    }


if __name__ == "__main__":
    run_benchmark(batch_size=16, num_steps=32, hidden_size=1400, iters=100)

"""Mergen-SNN — Eğitim Diverjansı Teşhis ve Deney Scripti.

50 adım boyunca her 10 adımda loss, gradient normları, spike oranları ve logits istatistiklerini izler.
"""

import argparse
import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mergen.data.char_dataset import CharDataset
from mergen.models.snn_lm_bpe import BpeSNNLM
from mergen.training.lm_trainer import get_cosine_schedule_with_warmup
from mergen.utils.device import device_name, get_device

# Windows konsolunda Unicode çıktı uyumluluğu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_diagnosis(
    mode_name: str,
    tie_weights: bool = True,
    optimizer_type: str = "sgd",
    lr: float = 0.0005,
    warmup_steps: int = 1000,
    num_steps: int = 50,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = get_device(prefer="cpu")

    data_path = Path("data/processed/bpe_train.bin")
    if not data_path.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_path}")

    dataset = CharDataset(data_path, block_size=32, stride=32)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    data_iter = iter(loader)

    vocab_size = 8192
    embed_dim = 384
    hidden_size = 384
    beta = 0.9

    model = BpeSNNLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        beta=beta,
        tie_weights=tie_weights,
    ).to(device)

    unique_params = sum(p.numel() for p in model.parameters())

    print("\n" + "=" * 80)
    print(f"TEŞHİS ÇALIŞTIRILIYOR: {mode_name}")
    print(f"Parametreler: tie_weights={tie_weights} (Tekil Param: {unique_params:,}) | opt={optimizer_type} | lr={lr}")
    print("=" * 80)

    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    if warmup_steps > 0:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=50000,
            min_lr_ratio=0.1,
        )
    else:
        scheduler = None

    model.train()
    history = []

    print(
        f"{'Step':>5} | {'Loss':>7} | {'EmbGrad':>8} | {'FC1Grad':>8} | {'FC2Grad':>8} | "
        f"{'OutGrad':>8} | {'Spk1':>6} | {'Spk2':>6} | {'LogitMean':>9} | {'LogitStd':>8} | {'LR':>9}"
    )
    print("-" * 105)

    for step in range(1, num_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits, spk1, spk2 = model(x, num_steps=32, return_spikes=True)

        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()

        # Gradient normlarını kaydet (clipping öncesi ham değerler)
        emb_grad_norm = model.embedding.weight.grad.norm().item() if model.embedding.weight.grad is not None else 0.0
        fc1_grad_norm = model.fc1.weight.grad.norm().item() if model.fc1.weight.grad is not None else 0.0
        fc2_grad_norm = model.fc2.weight.grad.norm().item() if model.fc2.weight.grad is not None else 0.0
        fc_out_grad_norm = model.fc_out.weight.grad.norm().item() if model.fc_out.weight.grad is not None else 0.0

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # İstatistikler
        with torch.no_grad():
            spk1_rate = (spk1 > 0).float().mean().item()
            spk2_rate = (spk2 > 0).float().mean().item()
            l_mean = logits.mean().item()
            l_std = logits.std().item()
            l_min = logits.min().item()
            l_max = logits.max().item()

        step_data = {
            "step": step,
            "loss": loss.item(),
            "emb_grad": emb_grad_norm,
            "fc1_grad": fc1_grad_norm,
            "fc2_grad": fc2_grad_norm,
            "fc_out_grad": fc_out_grad_norm,
            "spk1": spk1_rate,
            "spk2": spk2_rate,
            "logit_mean": l_mean,
            "logit_std": l_std,
            "logit_min": l_min,
            "logit_max": l_max,
            "lr": current_lr,
        }
        history.append(step_data)

        if step % 10 == 0 or step == 1 or step == num_steps:
            print(
                f"{step:5d} | {loss.item():7.4f} | {emb_grad_norm:8.4f} | {fc1_grad_norm:8.4f} | {fc2_grad_norm:8.4f} | "
                f"{fc_out_grad_norm:8.4f} | {spk1_rate:6.3f} | {spk2_rate:6.3f} | {l_mean:9.4f} | {l_std:8.4f} | {current_lr:9.6f}"
            )

    print("-" * 105)
    start_loss = history[0]["loss"]
    end_loss = history[-1]["loss"]
    loss_delta = end_loss - start_loss
    trend = "DÜŞÜYOR (İyi)" if loss_delta < 0 else "YÜKSELİYOR / DIVERGE (Kötü)"
    print(f"\nÖZET ({mode_name}):")
    print(f"  Başlangıç Loss (Step 1) : {start_loss:.4f}")
    print(f"  Bitiş Loss (Step {num_steps})   : {end_loss:.4f} (Fark: {loss_delta:+.4f}) -> {trend}")
    print(f"  Final Logit Dağılımı    : mean={end_loss:.2f}, std={history[-1]['logit_std']:.4f}, [min={history[-1]['logit_min']:.2f}, max={history[-1]['logit_max']:.2f}]")
    print(f"  Final Spike Oranları    : LIF1={history[-1]['spk1']:.3f}, LIF2={history[-1]['spk2']:.3f}")
    print(f"  Final Gradient Normları : Emb={history[-1]['emb_grad']:.4f}, FC1={history[-1]['fc1_grad']:.4f}, FC2={history[-1]['fc2_grad']:.4f}, Out={history[-1]['fc_out_grad']:.4f}")
    print("=" * 80 + "\n")

    return history


def main():
    parser = argparse.ArgumentParser(description="Mergen-SNN Eğitim Teşhis Aracı")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "baseline", "experiment_a", "experiment_b"],
        help="Çalıştırılacak teşhis modu",
    )
    parser.add_argument("--steps", type=int, default=50, help="Adım sayısı (varsayılan: 50)")
    args = parser.parse_args()

    modes = []
    if args.mode in ["all", "baseline"]:
        modes.append(("Baseline: Weight Tying + SGD(lr=0.0005, warmup=1000)", True, "sgd", 0.0005, 1000))
    if args.mode in ["all", "experiment_a"]:
        modes.append(("Deney A: No Weight Tying (Ayrı fc_out) + SGD(lr=0.0005, warmup=1000)", False, "sgd", 0.0005, 1000))
    if args.mode in ["all", "experiment_b"]:
        modes.append(("Deney B: Weight Tying + Adam(lr=0.0001, warmup=0)", True, "adam", 0.0001, 0))

    for name, tie, opt, lr, warmup in modes:
        run_diagnosis(
            mode_name=name,
            tie_weights=tie,
            optimizer_type=opt,
            lr=lr,
            warmup_steps=warmup,
            num_steps=args.steps,
        )


if __name__ == "__main__":
    main()

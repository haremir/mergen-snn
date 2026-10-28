"""Mergen-SNN — Stage 2 SNN Dil Modeli Eğitim Scripti."""

import math
import os
from pathlib import Path
import time
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import yaml

from mergen.data.char_dataset import CharDataset, load_charset
from mergen.models.snn_lm import CharSNNLM
from mergen.training.lm_trainer import eval_epoch_lm, train_epoch_lm
from mergen.utils.device import device_name, get_device
from mergen.utils.logger import get_logger

logger = get_logger("mergen.train_lm")


def load_config() -> dict:
    """default.yaml ve stage2_char.yaml yapılandırmalarını birleştirir."""
    configs_dir = Path("configs")
    default_cfg_path = configs_dir / "default.yaml"
    stage2_cfg_path = configs_dir / "stage2_char.yaml"

    config: Dict = {}
    if default_cfg_path.exists():
        with open(default_cfg_path, "r", encoding="utf-8") as f:
            config.update(yaml.safe_load(f) or {})

    if stage2_cfg_path.exists():
        with open(stage2_cfg_path, "r", encoding="utf-8") as f:
            stage2_cfg = yaml.safe_load(f) or {}
            for k, v in stage2_cfg.items():
                if isinstance(v, dict) and k in config and isinstance(config[k], dict):
                    config[k].update(v)
                else:
                    config[k] = v

    return config


def main():
    logger.info("==========================================")
    logger.info("Mergen-SNN — Aşama 2: SNN Dil Modeli Eğitimi")
    logger.info("==========================================")

    # 1. Config yükle
    cfg = load_config()

    prefer_device = cfg.get("device", {}).get("prefer", "cpu")
    device = get_device(prefer=prefer_device)
    dev_name = device_name(device)

    data_dir = Path(cfg.get("paths", {}).get("data_processed", "data/processed"))
    checkpoints_dir = Path(cfg.get("paths", {}).get("checkpoints", "checkpoints"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    block_size = int(cfg.get("data", {}).get("block_size", 32))
    batch_size = int(cfg.get("data", {}).get("batch_size", 32))
    num_workers = int(cfg.get("data", {}).get("num_workers", 0))

    embed_dim = int(cfg.get("model", {}).get("embed_dim", 128))
    hidden_size = int(cfg.get("model", {}).get("hidden_size", 256))
    beta = float(cfg.get("model", {}).get("beta", 0.9))
    num_steps = int(cfg.get("model", {}).get("num_steps", block_size))

    epochs = int(cfg.get("training", {}).get("epochs", 2))
    lr = float(cfg.get("training", {}).get("learning_rate", 0.01))
    momentum = float(cfg.get("training", {}).get("momentum", 0.9))
    foreach = bool(cfg.get("training", {}).get("foreach", False))
    max_steps_per_epoch = cfg.get("training", {}).get("max_steps_per_epoch", None)
    if max_steps_per_epoch is not None:
        max_steps_per_epoch = int(max_steps_per_epoch)
    log_every = int(cfg.get("training", {}).get("log_every", 500))
    resume_enabled = bool(cfg.get("training", {}).get("resume", True))

    # 2. Charset yükle ve vocab_size doğrula
    charset_path = data_dir / "charset.json"
    stoi, itos = load_charset(charset_path)
    real_vocab_size = len(stoi)
    cfg_vocab_size = cfg.get("model", {}).get("vocab_size", 460)

    if real_vocab_size != cfg_vocab_size:
        logger.warning(
            f"UYARI: charset.json vocab_size ({real_vocab_size}) != config vocab_size ({cfg_vocab_size}). "
            f"charset.json değeri ({real_vocab_size}) kullanılacak."
        )
    vocab_size = real_vocab_size

    # 3. Veri kümeleri ve DataLoader
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"

    logger.info(f"Veri yükleniyor: {train_bin} ve {val_bin}")
    train_dataset = CharDataset(train_bin, block_size=block_size)
    val_dataset = CharDataset(val_bin, block_size=block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # 4. Model oluştur
    model = CharSNNLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        beta=beta,
    ).to(device)

    # Parametre sayımı
    print("\n--- Model Katmanları ve Parametre Sayıları ---")
    total_params = 0
    for name, p in model.named_parameters():
        count = p.numel()
        total_params += count
        print(f"  {name:<25}: {count:>10,}  (şekil: {list(p.shape)})")
    print(f"Toplam Parametre Sayısı: {total_params:,}\n")

    logger.info(f"Cihaz: {dev_name} ({device})")
    logger.info(f"Model: vocab={vocab_size}, embed_dim={embed_dim}, hidden_size={hidden_size}, beta={beta}")
    logger.info(f"Eğitim: batch_size={batch_size}, num_steps={num_steps}, epochs={epochs}, lr={lr}")
    logger.info(f"Toplam adım/epoch: {max_steps_per_epoch or len(train_loader):,}")

    # 5. Optimizatör: SGD (foreach=False, DirectML/CPU uyumlu)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        foreach=foreach,
    )

    # Checkpoint Resume Kontrolü
    best_val_loss = float("inf")
    best_ckpt_path = checkpoints_dir / "lm_best.pt"

    if resume_enabled and best_ckpt_path.exists():
        logger.info(f"Mevcut checkpoint tespit edildi, yükleniyor: {best_ckpt_path}")
        try:
            checkpoint_data = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint_data["model_state_dict"])
            if "optimizer_state_dict" in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            best_val_loss = checkpoint_data.get("val_loss", float("inf"))
            prev_epoch = checkpoint_data.get("epoch", 0)
            logger.info(f"Checkpoint başarıyla yüklendi (Önceki en iyi val_loss: {best_val_loss:.4f}, epoch: {prev_epoch})")
        except Exception as e:
            logger.warning(f"Checkpoint yüklenirken hata oluştu (sıfırdan başlanıyor): {e}")

    # 6. Epoch Döngüsü
    train_losses = []
    val_losses = []

    total_training_start = time.time()

    for epoch in range(epochs):
        logger.info(f"--- Epoch {epoch + 1}/{epochs} Başladı ---")
        t0 = time.time()

        train_loss, train_elapsed, step_avg_sec, early_stopped = train_epoch_lm(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            num_steps=num_steps,
            max_steps=max_steps_per_epoch,
            log_every=log_every,
            patience_steps=5000,
        )

        val_loss, val_elapsed = eval_epoch_lm(
            model=model,
            loader=val_loader,
            device=device,
            num_steps=num_steps,
            max_steps=max_steps_per_epoch,
        )

        epoch_duration = time.time() - t0
        train_ppl = math.exp(min(train_loss, 20.0))
        val_ppl = math.exp(min(val_loss, 20.0))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        is_best = val_loss < best_val_loss
        best_tag = " | BEST" if is_best else ""

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} ppl={train_ppl:.2f} | "
            f"val_loss={val_loss:.4f} ppl={val_ppl:.2f} | "
            f"{epoch_duration:.0f}s{best_tag}"
        )

        # Epoch checkpoint kaydet
        ckpt_path = checkpoints_dir / f"lm_epoch{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "vocab_size": vocab_size,
                "embed_dim": embed_dim,
                "hidden_size": hidden_size,
                "beta": beta,
            },
            ckpt_path,
        )
        logger.info(f"Checkpoint kaydedildi: {ckpt_path}")

        # En iyi val loss kontrolü
        if is_best:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "vocab_size": vocab_size,
                    "embed_dim": embed_dim,
                    "hidden_size": hidden_size,
                    "beta": beta,
                },
                best_ckpt_path,
            )
            logger.info(f"Yeni en iyi model kaydedildi: {best_ckpt_path} (val_loss={val_loss:.4f})")

        if early_stopped:
            logger.warning("5000 adım boyunca plato nedeniyle eğitim erken durduruldu.")
            break

    # 7. Final model checkpoint kaydet
    final_ckpt_path = checkpoints_dir / "lm_final.pt"
    torch.save(
        {
            "epoch": len(train_losses),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_losses[-1] if train_losses else None,
            "val_loss": val_losses[-1] if val_losses else None,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_size": hidden_size,
            "beta": beta,
        },
        final_ckpt_path,
    )
    logger.info(f"Final model kaydedildi: {final_ckpt_path}")

    # 8. Eğitim eğrisi kaydet
    if len(train_losses) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker="o")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Mergen-SNN Stage 2: SNN Dil Modeli Eğitim Eğrisi")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        curve_path = checkpoints_dir / "lm_training_curve.png"
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Eğitim eğrisi kaydedildi: {curve_path}")

    total_time = time.time() - total_training_start
    logger.info(f"Tüm süreç tamamlandı. Toplam süre: {total_time:.1f}s")


if __name__ == "__main__":
    main()

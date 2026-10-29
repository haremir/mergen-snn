"""Mergen-SNN — SNN Dil Modeli Eğitim Scripti (Stage 2 Char & Stage 3 BPE)."""

import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import yaml

from mergen.data.char_dataset import CharDataset, load_charset
from mergen.models.snn_lm import CharSNNLM
from mergen.models.snn_lm_bpe import BpeSNNLM
from mergen.training.lm_trainer import (
    eval_epoch_lm,
    get_cosine_schedule_with_warmup,
    train_epoch_lm,
)
from mergen.utils.device import device_name, get_device
from mergen.utils.logger import get_logger

# Windows konsolunda Unicode çıktı uyumluluğu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = get_logger("mergen.train_lm")


def load_config(config_file: Optional[str] = None) -> dict:
    """default.yaml ve ilgili stage config dosyasını birleştirir."""
    configs_dir = Path("configs")
    default_cfg_path = configs_dir / "default.yaml"

    if config_file:
        stage_cfg_path = Path(config_file)
    else:
        # Öncelik Stage 3 BPE, yoksa Stage 2 Char
        stage3_path = configs_dir / "stage3_bpe.yaml"
        stage2_path = configs_dir / "stage2_char.yaml"
        stage_cfg_path = stage3_path if stage3_path.exists() else stage2_path

    config: Dict = {}
    if default_cfg_path.exists():
        with open(default_cfg_path, "r", encoding="utf-8") as f:
            config.update(yaml.safe_load(f) or {})

    if stage_cfg_path.exists():
        logger.info(f"Yapılandırma yükleniyor: {stage_cfg_path}")
        with open(stage_cfg_path, "r", encoding="utf-8") as f:
            stage_cfg = yaml.safe_load(f) or {}
            for k, v in stage_cfg.items():
                if isinstance(v, dict) and k in config and isinstance(config[k], dict):
                    config[k].update(v)
                else:
                    config[k] = v

    return config


def main():
    parser = argparse.ArgumentParser(description="Mergen-SNN Dil Modeli Eğitimi")
    parser.add_argument("--config", type=str, default=None, help="YAML yapılandırma dosyası yolu")
    parser.add_argument("--max_steps", type=int, default=None, help="Epoch başına maksimum adım (override)")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Checkpoint'ten devam et ('auto' veya özel dosya yolu)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    is_bpe = (
        cfg.get("model", {}).get("name") == "bpe_snn_lm"
        or cfg.get("data", {}).get("name") == "bpe_lm"
    )
    prefix = "bpe_lm" if is_bpe else "lm"
    stage_title = "Aşama 3: 10M BPE SNN (Mergen-Pico)" if is_bpe else "Aşama 2: Char-SNN"

    logger.info("==========================================")
    logger.info(f"Mergen-SNN — {stage_title} Eğitimi")
    logger.info("==========================================")

    prefer_device = cfg.get("device", {}).get("prefer", "cpu")
    device = get_device(prefer=prefer_device)
    dev_name = device_name(device)

    data_dir = Path(cfg.get("paths", {}).get("data_processed", "data/processed"))
    checkpoints_dir = Path(cfg.get("paths", {}).get("checkpoints", "checkpoints"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    block_size = int(cfg.get("data", {}).get("block_size", 128 if is_bpe else 32))
    batch_size = int(cfg.get("data", {}).get("batch_size", 32))
    num_workers = int(cfg.get("data", {}).get("num_workers", 0))

    embed_dim = int(cfg.get("model", {}).get("embed_dim", 512 if is_bpe else 128))
    hidden_size = int(cfg.get("model", {}).get("hidden_size", 1024 if is_bpe else 256))
    beta = float(cfg.get("model", {}).get("beta", 0.9))
    num_steps = int(cfg.get("model", {}).get("num_steps", block_size))

    epochs = int(cfg.get("training", {}).get("epochs", 3 if is_bpe else 2))
    lr = float(cfg.get("training", {}).get("learning_rate", 0.003 if is_bpe else 0.01))
    momentum = float(cfg.get("training", {}).get("momentum", 0.9))
    foreach = bool(cfg.get("training", {}).get("foreach", False))
    clip_grad_norm = float(cfg.get("training", {}).get("clip_grad_norm", 1.0))
    warmup_steps = int(cfg.get("training", {}).get("warmup_steps", 500))
    chars_per_token = float(cfg.get("training", {}).get("chars_per_token", 3.29 if is_bpe else 1.0))
    optimizer_name = str(cfg.get("training", {}).get("optimizer", "sgd")).lower()
    early_stop_patience = int(cfg.get("training", {}).get("early_stop_patience", 3000))

    max_steps_per_epoch = args.max_steps or cfg.get("training", {}).get("max_steps_per_epoch", None)
    if max_steps_per_epoch is not None:
        max_steps_per_epoch = int(max_steps_per_epoch)
    log_every = int(cfg.get("training", {}).get("log_every", 100))
    checkpoint_every = int(cfg.get("training", {}).get("checkpoint_every", 500))
    resume_cfg = cfg.get("training", {}).get("resume", False)
    resume_flag = args.resume if args.resume is not None else resume_cfg

    # Tokenizer ve Vocab Boyutu
    if is_bpe:
        from tokenizers import Tokenizer
        tok_path = Path(cfg.get("data", {}).get("tokenizer_path", data_dir / "tokenizer_bpe.json"))
        tokenizer = Tokenizer.from_file(str(tok_path))
        vocab_size = tokenizer.get_vocab_size()
        train_bin = Path(cfg.get("data", {}).get("train_bin", data_dir / "bpe_train.bin"))
        val_bin = Path(cfg.get("data", {}).get("val_bin", data_dir / "bpe_val.bin"))
    else:
        charset_path = data_dir / "charset.json"
        stoi, _ = load_charset(charset_path)
        vocab_size = len(stoi)
        train_bin = data_dir / "train.bin"
        val_bin = data_dir / "val.bin"

    logger.info(f"Veri kümeleri yükleniyor: {train_bin.name} ve {val_bin.name}")
    train_dataset = CharDataset(train_bin, block_size=block_size, stride=block_size)
    val_dataset = CharDataset(val_bin, block_size=block_size, stride=block_size)

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

    # Model oluştur
    tie_weights = bool(cfg.get("model", {}).get("tie_weights", True if embed_dim == hidden_size else False))
    if is_bpe:
        model = BpeSNNLM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            beta=beta,
            tie_weights=tie_weights,
        ).to(device)
    else:
        model = CharSNNLM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            beta=beta,
        ).to(device)

    # Parametre dökümü
    print("\n--- Model Katmanları ve Parametre Sayıları ---")
    total_params = 0
    seen_params = set()
    for name, p in model.named_parameters(remove_duplicate=False):
        count = p.numel()
        if p not in seen_params:
            total_params += count
            seen_params.add(p)
            print(f"  {name:<25}: {count:>12,}  (şekil: {list(p.shape)})")
        else:
            print(f"  {name:<25}: {count:>12,}  (şekil: {list(p.shape)}) [Tied / Paylaşımlı]")
    print(f"Toplam Parametre Sayısı: {total_params:,}\n")

    logger.info(f"Cihaz: {dev_name} ({device})")
    logger.info(f"Model: {model.__class__.__name__} | vocab={vocab_size}, embed={embed_dim}, hidden={hidden_size}")
    logger.info(f"Eğitim: batch_size={batch_size}, num_steps={num_steps}, epochs={epochs}, lr={lr}")
    logger.info(f"Adım Sınırı (max_steps_per_epoch): {max_steps_per_epoch}")

    # Optimizatör: Adam veya SGD
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            foreach=foreach,
        )

    # Öğrenme Oranı Planlayıcısı (Warmup + Cosine Decay)
    steps_per_epoch = max_steps_per_epoch if max_steps_per_epoch is not None else len(train_loader)
    total_training_steps = max(epochs * steps_per_epoch, 1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
        min_lr_ratio=0.1,
    )
    logger.info(f"LR Planlayıcı: Warmup={warmup_steps} adım, Toplam Adım={total_training_steps}")

    # Resume kontrolü
    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_ckpt_path = checkpoints_dir / f"{prefix}_best.pt"

    if resume_flag:
        resume_target = None
        # 1. Eğer doğrudan bir dosya yolu verilmişse
        if isinstance(resume_flag, str) and resume_flag != "auto" and Path(resume_flag).exists():
            resume_target = Path(resume_flag)
        else:
            # 2. Önce en yüksek step checkpoint'ini ara (checkpoints/{prefix}_step*.pt)
            step_files = sorted(
                checkpoints_dir.glob(f"{prefix}_step*.pt"),
                key=lambda f: int(f.stem.replace(f"{prefix}_step", ""))
                if f.stem.replace(f"{prefix}_step", "").isdigit()
                else 0,
            )
            if step_files:
                resume_target = step_files[-1]
            else:
                # 3. Epoch checkpoint'lerini ara (checkpoints/{prefix}_epoch*.pt)
                epoch_files = sorted(
                    checkpoints_dir.glob(f"{prefix}_epoch*.pt"),
                    key=lambda f: int(f.stem.replace(f"{prefix}_epoch", ""))
                    if f.stem.replace(f"{prefix}_epoch", "").isdigit()
                    else 0,
                )
                if epoch_files:
                    resume_target = epoch_files[-1]
                elif (checkpoints_dir / f"{prefix}_best.pt").exists():
                    resume_target = checkpoints_dir / f"{prefix}_best.pt"

        if resume_target and resume_target.exists():
            logger.info(f"Resume checkpoint yükleniyor: {resume_target}")
            try:
                ckpt_data = torch.load(resume_target, map_location=device, weights_only=False)
                model.load_state_dict(ckpt_data["model_state_dict"])
                if "optimizer_state_dict" in ckpt_data and ckpt_data["optimizer_state_dict"] is not None:
                    optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])

                best_val_loss = ckpt_data.get("best_val_loss", ckpt_data.get("val_loss", float("inf")))
                if "rng_state" in ckpt_data and ckpt_data["rng_state"] is not None:
                    torch.set_rng_state(ckpt_data["rng_state"])

                # Adım ve epoch hesaplama
                if "step" in ckpt_data and ckpt_data["step"] is not None:
                    global_step = int(ckpt_data["step"])
                    start_epoch = global_step // steps_per_epoch
                    start_step_in_epoch = global_step % steps_per_epoch
                elif "epoch" in ckpt_data and ckpt_data["epoch"] is not None:
                    completed_epoch = int(ckpt_data["epoch"])
                    start_epoch = completed_epoch
                    start_step_in_epoch = 0
                    global_step = completed_epoch * steps_per_epoch

                # Scheduler durumu
                if "scheduler_state_dict" in ckpt_data and ckpt_data["scheduler_state_dict"] is not None:
                    scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
                else:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        for _ in range(global_step):
                            scheduler.step()

                logger.info(
                    f"Resume başarılı! Kaldığı yer: Epoch {start_epoch + 1}/{epochs}, "
                    f"Epoch-İçi Adım {start_step_in_epoch}/{steps_per_epoch}, Toplam Adım {global_step}, "
                    f"Önceki best_val_loss={best_val_loss:.4f}"
                )
            except Exception as e:
                logger.warning(f"Resume checkpoint yüklenemedi: {e}")
        else:
            logger.info("Resume için uygun checkpoint bulunamadı, eğitim sıfırdan başlıyor.")

    train_losses = []
    val_losses = []
    total_training_start = time.time()

    for epoch in range(start_epoch, epochs):
        current_start_step = start_step_in_epoch if epoch == start_epoch else 0
        logger.info(f"--- Epoch {epoch + 1}/{epochs} Başladı (Başlangıç Adımı: {current_start_step}/{steps_per_epoch}) ---")
        t0 = time.time()

        train_loss, train_bpb, train_elapsed, step_avg_sec, early_stopped, steps_done = train_epoch_lm(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            num_steps=num_steps,
            max_steps=max_steps_per_epoch,
            log_every=log_every,
            patience_steps=early_stop_patience,
            clip_grad_norm=clip_grad_norm,
            scheduler=scheduler,
            chars_per_token=chars_per_token,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=checkpoints_dir,
            checkpoint_prefix=prefix,
            epoch_idx=epoch,
            global_step_offset=global_step,
            best_val_loss=best_val_loss,
            start_step_in_epoch=current_start_step,
            is_bpe=is_bpe,
        )
        global_step += steps_done

        val_loss, val_bpb, val_elapsed = eval_epoch_lm(
            model=model,
            loader=val_loader,
            device=device,
            num_steps=num_steps,
            max_steps=max_steps_per_epoch,
            chars_per_token=chars_per_token,
        )

        epoch_duration = time.time() - t0
        train_ppl = math.exp(min(train_loss, 20.0))
        val_ppl = math.exp(min(val_loss, 20.0))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        is_best = val_loss < best_val_loss
        best_tag = " | BEST" if is_best else ""

        logger.info(
            f"Epoch {epoch + 1}/{epochs} Tamamlandı | "
            f"train_loss={train_loss:.4f} ppl={train_ppl:.2f} bpb={train_bpb:.3f} | "
            f"val_loss={val_loss:.4f} ppl={val_ppl:.2f} bpb={val_bpb:.3f} | "
            f"{epoch_duration:.0f}s (step_ort={step_avg_sec * 1000:.1f}ms){best_tag}"
        )

        # Checkpoint kaydet
        ckpt_path = checkpoints_dir / f"{prefix}_epoch{epoch + 1}.pt"
        ckpt_payload = {
            "epoch": epoch + 1,
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "rng_state": torch.get_rng_state(),
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_size": hidden_size,
            "beta": beta,
            "is_bpe": is_bpe,
            "tie_weights": getattr(model, "tie_weights", False),
        }
        torch.save(ckpt_payload, ckpt_path)
        logger.info(f"Checkpoint kaydedildi: {ckpt_path}")

        if is_best:
            best_val_loss = val_loss
            ckpt_payload["best_val_loss"] = best_val_loss
            torch.save(ckpt_payload, best_ckpt_path)
            logger.info(f"Yeni en iyi model kaydedildi: {best_ckpt_path} (val_loss={val_loss:.4f})")

        if early_stopped:
            logger.warning("Plato tespiti nedeniyle eğitim erken durduruldu.")
            break

    # Final model checkpoint
    final_ckpt_path = checkpoints_dir / f"{prefix}_final.pt"
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
            "is_bpe": is_bpe,
        },
        final_ckpt_path,
    )
    logger.info(f"Final model kaydedildi: {final_ckpt_path}")

    # Eğitim eğrisi kaydet
    if len(train_losses) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker="o")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"Mergen-SNN {stage_title}: Eğitim Eğrisi")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        curve_path = checkpoints_dir / f"{prefix}_training_curve.png"
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Eğitim eğrisi kaydedildi: {curve_path}")

    total_time = time.time() - total_training_start
    logger.info(f"Tüm süreç tamamlandı. Toplam süre: {total_time:.1f}s")


if __name__ == "__main__":
    main()

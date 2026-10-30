"""Mergen-SNN — Stage 4: Small SNN Dil Modeli (~9M-20M Parametre) Eğitim Scripti.

Google Colab ve Drive entegrasyonu, atomik checkpoint kaydı ve otomatik resume desteği.
"""

import argparse
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from mergen.models.snn_lm_bpe import BpeSNNLM
from mergen.utils.device import device_name, get_device
from mergen.utils.logger import get_logger

# Windows Unicode konsol uyumluluğu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# CPU thread ayarları
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "6")
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "6")

logger = get_logger("mergen.train_small")


def atomic_save(state: dict, dest_path: Path):
    """Drive FUSE gecikmelerine ve elektrik/oturum kesintilerine karşı atomik checkpoint kaydı."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(f".tmp_{dest_path.name}")
    torch.save(state, tmp_path)
    if hasattr(os, "sync"):
        try:
            os.sync()
        except Exception:
            pass
    os.replace(tmp_path, dest_path)


class BinaryTokenDataset(Dataset):
    """NumPy bellek haritalı (memmap) BPE binary veri kümesi."""

    def __init__(self, bin_path: Path, block_size: int = 32):
        super().__init__()
        if not bin_path.exists():
            raise FileNotFoundError(f"Binary veri dosyası bulunamadı: {bin_path}")
        self.block_size = block_size
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        self.num_samples = len(self.data) - block_size

    def __len__(self) -> int:
        return max(0, self.num_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = torch.from_numpy(self.data[idx : idx + self.block_size + 1].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_config(config_file: Optional[str] = None) -> dict:
    """default.yaml ve stage4_small.yaml yapılandırma dosyalarını birleştirir."""
    configs_dir = Path("configs")
    default_cfg_path = configs_dir / "default.yaml"
    stage4_path = Path(config_file) if config_file else configs_dir / "stage4_small.yaml"

    config: Dict = {}
    if default_cfg_path.exists():
        with open(default_cfg_path, "r", encoding="utf-8") as f:
            config.update(yaml.safe_load(f) or {})

    if stage4_path.exists():
        logger.info(f"Yapılandırma yükleniyor: {stage4_path}")
        with open(stage4_path, "r", encoding="utf-8") as f:
            stage_cfg = yaml.safe_load(f) or {}
            for k, v in stage_cfg.items():
                if isinstance(v, dict) and k in config and isinstance(config[k], dict):
                    config[k].update(v)
                else:
                    config[k] = v

    return config


def find_latest_checkpoint(ckpt_dir: Path, prefix: str = "small_lm") -> Optional[Path]:
    """Checkpoint dizininde en son adım veya epoch checkpoint'ini bulur."""
    if not ckpt_dir.exists():
        return None

    step_files: List[Tuple[int, Path]] = []
    epoch_files: List[Tuple[int, Path]] = []

    for f in ckpt_dir.glob("*.pt"):
        step_match = re.search(rf"{prefix}_step(\d+)\.pt$", f.name)
        if step_match:
            step_files.append((int(step_match.group(1)), f))
            continue
        epoch_match = re.search(rf"{prefix}_epoch(\d+)\.pt$", f.name)
        if epoch_match:
            epoch_files.append((int(epoch_match.group(1)), f))

    if step_files:
        step_files.sort(key=lambda x: x[0], reverse=True)
        return step_files[0][1]

    if epoch_files:
        epoch_files.sort(key=lambda x: x[0], reverse=True)
        return epoch_files[0][1]

    best_path = ckpt_dir / f"{prefix}_best.pt"
    if best_path.exists():
        return best_path

    return None


def prune_old_step_checkpoints(ckpt_dir: Path, prefix: str = "small_lm", keep_last: int = 3):
    """Sadece son `keep_last` adet step checkpoint'ini tutar, eskileri siler."""
    step_files: List[Tuple[int, Path]] = []
    for f in ckpt_dir.glob("*.pt"):
        m = re.search(rf"{prefix}_step(\d+)\.pt$", f.name)
        if m:
            step_files.append((int(m.group(1)), f))

    if len(step_files) > keep_last:
        step_files.sort(key=lambda x: x[0], reverse=True)
        for _, old_file in step_files[keep_last:]:
            try:
                old_file.unlink()
                logger.debug(f"Eski step checkpoint silindi: {old_file.name}")
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Mergen-SNN Stage 4 Small Model Eğitimi")
    parser.add_argument("--config", type=str, default="configs/stage4_small.yaml", help="Yapılandırma dosyası")
    parser.add_argument("--max_steps", type=int, default=None, help="Maksimum adım sınırı (test ve doğrulama için)")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Checkpoint'ten devam et ('auto' veya özel yol)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    prefix = "small_lm"
    logger.info("==========================================")
    logger.info("Mergen-SNN — Aşama 4: Small SNN Modeli Eğitimi")
    logger.info("==========================================")

    prefer_device = cfg.get("device", {}).get("prefer", "cpu")
    device = get_device(prefer=prefer_device)
    dev_name = device_name(device)

    data_dir = Path(cfg.get("paths", {}).get("data_processed", "data/processed"))
    train_bin = Path(cfg.get("data", {}).get("train_bin", data_dir / "bpe_train.bin"))
    val_bin = Path(cfg.get("data", {}).get("val_bin", data_dir / "bpe_val.bin"))

    block_size = int(cfg.get("data", {}).get("block_size", 32))
    batch_size = int(cfg.get("data", {}).get("batch_size", 128))

    logger.info(f"Veri yükleniyor: {train_bin.name} ve {val_bin.name}")
    train_dataset = BinaryTokenDataset(train_bin, block_size=block_size)
    val_dataset = BinaryTokenDataset(val_bin, block_size=block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    # Model parametreleri
    vocab_size = int(cfg.get("model", {}).get("vocab_size", 8192))
    embed_dim = int(cfg.get("model", {}).get("embed_dim", 512))
    hidden_size = int(cfg.get("model", {}).get("hidden_size", 512))
    beta = float(cfg.get("model", {}).get("beta", 0.9))
    num_steps = int(cfg.get("model", {}).get("num_steps", block_size))
    tie_weights = bool(cfg.get("model", {}).get("tie_weights", False))

    model = BpeSNNLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        beta=beta,
        tie_weights=tie_weights,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("\n--- Model Katmanları ve Parametre Sayıları ---")
    for name, p in model.named_parameters():
        print(f"  {name:<24} : {p.numel():>12,}  (Şekil: {list(p.shape)})")
    print(f"Toplam Parametre Sayısı: {total_params:,}\n")

    # Eğitim parametreleri
    epochs = int(cfg.get("training", {}).get("epochs", 3))
    lr = float(cfg.get("training", {}).get("learning_rate", 0.0002))
    warmup_steps = int(cfg.get("training", {}).get("warmup_steps", 500))
    clip_grad_norm = float(cfg.get("training", {}).get("clip_grad_norm", 1.0))
    checkpoint_every = int(cfg.get("training", {}).get("checkpoint_every", 500))
    checkpoint_dir = Path(cfg.get("training", {}).get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_training_steps = len(train_loader) * epochs
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume mantığı
    resume_flag = args.resume or cfg.get("training", {}).get("resume")
    resume_ckpt_path = None
    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if resume_flag:
        if str(resume_flag).lower() == "auto":
            resume_ckpt_path = find_latest_checkpoint(checkpoint_dir, prefix=prefix)
        else:
            p = Path(resume_flag)
            if p.exists():
                resume_ckpt_path = p

    if resume_ckpt_path and resume_ckpt_path.exists():
        logger.info(f"Resume checkpoint yükleniyor: {resume_ckpt_path}")
        ckpt_data = torch.load(resume_ckpt_path, map_location=device, weights_only=False)

        # Parametre şekil uyumluluğu kontrolü
        ckpt_embed = ckpt_data.get("embed_dim", embed_dim)
        ckpt_hidden = ckpt_data.get("hidden_size", hidden_size)
        if ckpt_embed == embed_dim and ckpt_hidden == hidden_size:
            model.load_state_dict(ckpt_data["model_state_dict"])
            if "optimizer_state_dict" in ckpt_data and ckpt_data["optimizer_state_dict"]:
                try:
                    optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
                except Exception as e:
                    logger.warning(f"Optimizer state yüklenemedi: {e}")

            if "scheduler_state_dict" in ckpt_data and ckpt_data["scheduler_state_dict"]:
                try:
                    scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
                except Exception as e:
                    logger.warning(f"Scheduler state yüklenemedi: {e}")

            if "rng_state" in ckpt_data and ckpt_data["rng_state"] is not None:
                torch.set_rng_state(ckpt_data["rng_state"].cpu())

            global_step = ckpt_data.get("step", 0)
            best_val_loss = ckpt_data.get("best_val_loss", float("inf"))
            saved_epoch = ckpt_data.get("epoch", 1)
            steps_per_epoch = len(train_loader)
            start_epoch = max(0, saved_epoch - 1)
            start_step_in_epoch = global_step % steps_per_epoch if steps_per_epoch > 0 else 0

            logger.info(
                f"Resume başarılı! Kaldığı yer: Epoch {start_epoch + 1}/{epochs}, "
                f"Adım {start_step_in_epoch}/{steps_per_epoch}, Toplam Adım {global_step}, "
                f"best_val_loss={best_val_loss:.4f}"
            )
        else:
            logger.warning(f"Checkpoint model boyutları uyuşmuyor ({ckpt_embed}x{ckpt_hidden} vs {embed_dim}x{hidden_size}). Sıfırdan başlanıyor.")

    # Eğitim döngüsü
    max_steps = args.max_steps or cfg.get("training", {}).get("max_steps_per_epoch")

    logger.info(f"Eğitim başlıyor: {epochs} epoch, cihaz: {dev_name}")
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        step_in_epoch = 0

        logger.info(f"--- Epoch {epoch + 1}/{epochs} Başladı (Başlangıç Adımı: {start_step_in_epoch}) ---")

        for x, y in train_loader:
            if start_step_in_epoch > 0 and step_in_epoch < start_step_in_epoch:
                step_in_epoch += 1
                continue

            step_in_epoch += 1
            global_step += 1
            step_t0 = time.time()

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, _, spk2 = model(x, num_steps=num_steps, return_spikes=True)

            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()

            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            optimizer.step()
            scheduler.step()

            step_dt = time.time() - step_t0
            cur_lr = optimizer.param_groups[0]["lr"]
            n_tokens = y.numel()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens

            spk2_ratio = (spk2 > 0).float().mean().item()
            ppl = math.exp(min(loss.item(), 20.0))
            bpb = loss.item() / math.log(2.0)

            if global_step % 500 == 0 or (max_steps and global_step <= max_steps):
                print(
                    f"Epoch {epoch + 1}/{epochs} | Adım {step_in_epoch}/{len(train_loader)} (Top {global_step}) "
                    f"| Loss: {loss.item():.4f} | PPL: {ppl:6.2f} | BPB: {bpb:.3f} | Spk2: {spk2_ratio*100:4.1f}% "
                    f"| LR: {cur_lr:.6f} | {step_dt:.2f}s/step",
                    flush=True,
                )

            # Step-level atomik checkpoint
            if checkpoint_every and (global_step % checkpoint_every == 0 or (max_steps and global_step == max_steps)):
                step_ckpt_path = checkpoint_dir / f"{prefix}_step{global_step}.pt"
                save_payload = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "rng_state": torch.get_rng_state(),
                    "train_loss": loss.item(),
                    "vocab_size": vocab_size,
                    "embed_dim": embed_dim,
                    "hidden_size": hidden_size,
                    "beta": beta,
                    "tie_weights": tie_weights,
                }
                atomic_save(save_payload, step_ckpt_path)
                logger.info(f"Step checkpoint kaydedildi: {step_ckpt_path.name}")
                prune_old_step_checkpoints(checkpoint_dir, prefix=prefix, keep_last=3)

            if max_steps and global_step >= max_steps:
                logger.info(f"Maksimum adım sınırına ({max_steps}) ulaşıldı. Eğitim durduruluyor.")
                return

        # Epoch sonu doğrulama
        start_step_in_epoch = 0
        avg_train_loss = epoch_loss / max(epoch_tokens, 1)

        model.eval()
        val_loss = 0.0
        val_tokens = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                vy = vy.to(device)
                v_logits = model(vx, num_steps=num_steps)
                vl = F.cross_entropy(v_logits.reshape(-1, vocab_size), vy.reshape(-1))
                val_loss += vl.item() * vy.numel()
                val_tokens += vy.numel()

        avg_val_loss = val_loss / max(val_tokens, 1)
        val_ppl = math.exp(min(avg_val_loss, 20.0))
        val_bpb = avg_val_loss / math.log(2.0)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} Tamamlandı | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:.2f}, BPB: {val_bpb:.3f})"
        )

        epoch_ckpt_path = checkpoint_dir / f"{prefix}_epoch{epoch + 1}.pt"
        payload = {
            "epoch": epoch + 1,
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "best_val_loss": best_val_loss,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_size": hidden_size,
            "beta": beta,
            "tie_weights": tie_weights,
        }
        atomic_save(payload, epoch_ckpt_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            payload["best_val_loss"] = best_val_loss
            atomic_save(payload, checkpoint_dir / f"{prefix}_best.pt")
            logger.info(f"Yeni en iyi model kaydedildi: {prefix}_best.pt (val_loss={best_val_loss:.4f})")

    total_time = time.time() - start_time
    logger.info(f"Eğitim tamamlandı! Toplam süre: {total_time:.1f}s")


if __name__ == "__main__":
    main()

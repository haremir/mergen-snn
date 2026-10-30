"""Mergen-SNN — SFT (Supervised Fine-Tuning) Eğitim Scripti.

Modeli checkpoints/bpe_lm_best.pt üzerinden yükler, yeni özel chat tokenları ile
genişletir ve data/sft/ altındaki maskelenmiş chat verisi ile eğitir.
"""

import argparse
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mergen.models.snn_lm_bpe import BpeSNNLM
from mergen.utils.logger import get_logger

# Windows Unicode konsol uyumluluğu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = get_logger("mergen.train_sft")


class SFTBinaryDataset(Dataset):
    """Bellek haritalı veya bellek içi SFT binary veri kümesi."""

    def __init__(self, train_bin: Path, mask_bin: Path, seq_len: int = 256):
        super().__init__()
        if not train_bin.exists():
            raise FileNotFoundError(f"Eğitim ikili dosyası bulunamadı: {train_bin}")
        if not mask_bin.exists():
            raise FileNotFoundError(f"Maske ikili dosyası bulunamadı: {mask_bin}")

        self.seq_len = seq_len
        self.inputs = np.fromfile(str(train_bin), dtype=np.uint16).reshape(-1, seq_len)
        self.masks = np.fromfile(str(mask_bin), dtype=np.uint8).reshape(-1, seq_len - 1)

        assert len(self.inputs) == len(self.masks), (
            f"Örnek sayısı uyuşmazlığı: inputs={len(self.inputs)}, masks={len(self.masks)}"
        )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        inp = torch.from_numpy(self.inputs[idx].astype(np.int64))
        mask = torch.from_numpy(self.masks[idx].astype(np.float32))

        # Causal LM: x = inp[:-1], y = inp[1:]
        x = inp[:-1]
        y = inp[1:]
        return x, y, mask


def main():
    parser = argparse.ArgumentParser(description="Mergen-SNN SFT Eğitimi")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/bpe_lm_best.pt", help="Başlangıç checkpoint'i")
    parser.add_argument("--train_bin", type=str, default="data/sft/sft_train.bin", help="SFT girdi verisi")
    parser.add_argument("--mask_bin", type=str, default="data/sft/sft_mask.bin", help="SFT kayıp maskesi")
    parser.add_argument("--meta_json", type=str, default="data/sft/sft_meta.json", help="SFT meta verisi")
    parser.add_argument("--out_checkpoint", type=str, default="checkpoints/mergen_sft.pt", help="Kayıt checkpoint'i")
    parser.add_argument("--epochs", type=int, default=3, help="Epoch sayısı (Varsayılan: 3)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch boyutu (Varsayılan: 64)")
    parser.add_argument("--seq_len", type=int, default=256, help="Dizi uzunluğu (Varsayılan: 256)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Öğrenme oranı (Varsayılan: 0.0001)")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup adım sayısı (Varsayılan: 100)")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradyan kırpma normu")
    parser.add_argument("--device", type=str, default="cpu", help="Cihaz (cpu / directml)")
    args = parser.parse_args()

    # OMP ve MKL thread ayarları
    os.environ["OMP_NUM_THREADS"] = "6"
    os.environ["MKL_NUM_THREADS"] = "6"

    device = torch.device(args.device)

    print("=" * 70)
    print("MERGEN-SNN — AŞAMA 4: SFT (SUPERVISED FINE-TUNING) EĞİTİMİ")
    print(f"Başlangıç Modeli : {args.checkpoint}")
    print(f"Epoch / Batch    : {args.epochs} Epoch | Batch: {args.batch_size} | Seq: {args.seq_len}")
    print(f"Öğrenme Oranı    : {args.lr} (Warmup: {args.warmup_steps} adım)")
    print(f"Cihaz            : {device}")
    print("=" * 70)

    # 1. Veri kümesini yükle
    dataset = SFTBinaryDataset(
        train_bin=Path(args.train_bin),
        mask_bin=Path(args.mask_bin),
        seq_len=args.seq_len,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    total_steps = len(loader) * args.epochs

    print(f"Veri Kümesi: {len(dataset)} örnek | {len(loader)} batch/epoch | Toplam {total_steps} adım")

    # 2. Modeli yükle ve genişlet
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    old_vocab = checkpoint.get("vocab_size", 8192)
    embed_dim = checkpoint.get("embed_dim", 384)
    hidden_size = checkpoint.get("hidden_size", 384)
    beta = checkpoint.get("beta", 0.9)
    tie_weights = checkpoint.get("tie_weights", False)

    # Model ilklendirme (eski vocab_size ile)
    model = BpeSNNLM(
        vocab_size=old_vocab,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        beta=beta,
        tie_weights=tie_weights,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Özel tokenlar için vocab genişletme: 8192 -> 8196
    new_vocab = 8196
    print(f"Model token dağarcığı genişletiliyor: {old_vocab} -> {new_vocab}...")
    model.resize_token_embeddings(new_vocab)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Toplam Parametre Sayısı: {total_params:,}")

    # 3. Optimizer ve Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Warmup + Cosine Scheduler
    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 4. Eğitim Döngüsü
    model.train()
    start_time = time.time()
    global_step = 0
    epoch_losses = []
    final_step_loss = 0.0

    print("\n--- SFT Eğitimi Başlatılıyor ---")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        running_tokens = 0
        running_spk2 = 0.0
        batch_count = 0

        for x, y, mask in loader:
            global_step += 1
            batch_count += 1
            step_t0 = time.time()

            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            num_steps = x.size(1)  # 255
            optimizer.zero_grad()

            logits, _, spk2 = model(x, num_steps=num_steps, return_spikes=True)

            # Kayıp hesaplama (yalnızca asistan yanıtı tokenları)
            loss_unreduced = F.cross_entropy(
                logits.reshape(-1, new_vocab),
                y.reshape(-1),
                reduction="none",
            )
            mask_flat = mask.reshape(-1)
            active_tokens = mask_flat.sum()

            loss = (loss_unreduced * mask_flat).sum() / active_tokens.clamp(min=1.0)
            loss.backward()

            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            optimizer.step()
            scheduler.step()

            step_dt = time.time() - step_t0
            cur_lr = optimizer.param_groups[0]["lr"]

            # İstatistikler
            l_val = loss.item()
            final_step_loss = l_val
            running_loss += l_val * active_tokens.item()
            running_tokens += int(active_tokens.item())

            spk_ratio = (spk2 > 0).float().mean().item()
            running_spk2 += spk_ratio

            ppl = math.exp(min(l_val, 20.0))
            bpb = l_val / math.log(2.0)

            print(
                f"Epoch {epoch}/{args.epochs} | Adım {batch_count}/{len(loader)} (Top {global_step}/{total_steps}) "
                f"| Loss: {l_val:.4f} | PPL: {ppl:6.2f} | BPB: {bpb:.3f} | Spk2: {spk_ratio*100:4.1f}% "
                f"| LR: {cur_lr:.6f} | {step_dt:.2f}s/step",
                flush=True,
            )

        epoch_loss = running_loss / max(running_tokens, 1)
        epoch_ppl = math.exp(min(epoch_loss, 20.0))
        epoch_bpb = epoch_loss / math.log(2.0)
        epoch_spk2 = running_spk2 / max(batch_count, 1)
        epoch_dt = time.time() - epoch_start
        epoch_losses.append(epoch_loss)

        print("-" * 70)
        print(
            f"Epoch {epoch}/{args.epochs} Tamamlandı ({epoch_dt:.1f}s) "
            f"| Ort. Loss: {epoch_loss:.4f} | PPL: {epoch_ppl:.2f} | BPB: {epoch_bpb:.3f} "
            f"| Ort. Spk2: {epoch_spk2*100:.1f}%",
            flush=True,
        )
        print("-" * 70)

    total_duration = time.time() - start_time

    # 5. Checkpoint Kaydet
    out_path = Path(args.out_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "epoch": args.epochs,
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": final_step_loss,
        "epoch_losses": epoch_losses,
        "vocab_size": new_vocab,
        "embed_dim": embed_dim,
        "hidden_size": hidden_size,
        "beta": beta,
        "is_bpe": True,
        "tie_weights": tie_weights,
    }
    torch.save(save_dict, out_path)

    print("\n" + "=" * 70)
    print("SFT EĞİTİMİ BAŞARIYLA TAMAMLANDI")
    print(f"Toplam Süre        : {total_duration:.1f} saniye ({total_duration/60:.2f} dakika)")
    print(f"Final Adım Kaybı   : {final_step_loss:.4f}")
    print(f"Final Epoch Kaybı  : {epoch_losses[-1]:.4f}")
    print(f"Model Kaydedildi   : {out_path} ({os.path.getsize(out_path):,} bayt)")
    print("=" * 70)


if __name__ == "__main__":
    main()

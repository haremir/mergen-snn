"""Mergen-SNN — Colab 50M Uçtan Uca Otomatik Pipeline (Veri Üretimi + Eğitim).

1. Colab üzerinde açık Hugging Face kaynaklarından (Code, Türkçe, FineWeb, ArXiv)
   homojen round-robin akış ile bpe_train_v3.bin ve bpe_val_v3.bin üretir (~500M token).
2. Veri hazır olduğu anda Tesla T4 GPU üzerinde 50M SNN dil modeli eğitimini
   otomatik olarak başlatır ve /content/mergen-snn/runs/stage6_50m.log dosyasına yazar.
"""

import os
from pathlib import Path
import subprocess
import sys
import time

def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def main():
    repo_dir = Path("/content/mergen-snn")
    os.chdir(repo_dir)
    os.makedirs("runs", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    train_bin = repo_dir / "data" / "processed" / "bpe_train_v3.bin"
    val_bin = repo_dir / "data" / "processed" / "bpe_val_v3.bin"

    # Adım 1: Veri Kümesi Kontrolü ve Üretimi
    # En az 200 MB (100M token) veri yoksa üret
    min_size_bytes = 200 * 1024 * 1024
    if not train_bin.exists() or train_bin.stat().st_size < min_size_bytes:
        log(">>> [ADIM 1] bpe_train_v3.bin bulunamadı veya boyutu küçük. Streaming veri üretimi başlatılıyor...")
        data_cmd = [
            sys.executable, "scripts/prepare_bpe_data_massive.py",
            "--train-out", str(train_bin),
            "--val-out", str(val_bin),
            "--max-tokens", "500000000"
        ]
        log(f"Çalıştırılan komut: {' '.join(data_cmd)}")
        with open("runs/data_prep.log", "w", encoding="utf-8") as f_prep:
            p = subprocess.run(data_cmd, stdout=f_prep, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                log(f"❌ HATA: Veri üretimi başarısız oldu (exit code: {p.returncode})! Logu kontrol edin.")
                sys.exit(p.returncode)
        sz_mb = train_bin.stat().st_size / (1024 * 1024) if train_bin.exists() else 0
        log(f">>> [OK] Veri üretimi tamamlandı! Train boyutu: {sz_mb:.1f} MB")
    else:
        sz_mb = train_bin.stat().st_size / (1024 * 1024)
        log(f">>> [OK] Mevcut bpe_train_v3.bin doğrulandı ({sz_mb:.1f} MB). Veri üretimi atlanıyor.")

    # Adım 2: Eğitimi Başlat
    log(">>> [ADIM 2] 50M SNN Dil Modeli Eğitimi Başlatılıyor...")
    train_log = repo_dir / "runs" / "stage6_50m.log"
    train_cmd = [
        sys.executable, "scripts/train_lm.py",
        "--config", "configs/stage6_50m_16k.yaml",
        "--resume", "auto"
    ]
    log(f"Eğitim komutu: {' '.join(train_cmd)}")
    with open(train_log, "a", encoding="utf-8") as f_train:
        proc = subprocess.Popen(train_cmd, stdout=f_train, stderr=subprocess.STDOUT)

    log(f"🚀 Eğitim süreci arka planda başarıyla başlatıldı! PID: {proc.pid}")
    log(f"Log Dosyası: {train_log}")

if __name__ == "__main__":
    main()

import os
import shutil
from pathlib import Path

# 0.1 Başlangıç
total, used, free = shutil.disk_usage("C:\\")
print(f"Başlangıç Disk Durumu: Toplam: {total//(2**30)} GB | Kullanılan: {used//(2**30)} GB | Boş: {free//(2**30)} GB")
if free // (2**30) < 15:
    print("UYARI: Boş disk alanı 15 GB'ın altında!")

# 0.2 Ham Veri Temizliği (data/raw/cs_corpus/)
raw_dir = Path("data/raw/cs_corpus")
extensions_to_delete = {".pdf", ".html", ".htm", ".zip", ".gz", ".tar", ".7z", ".docx", ".epub"}
deleted_raw_size = 0
deleted_raw_count = 0
if raw_dir.exists():
    for f in list(raw_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in extensions_to_delete:
            try:
                sz = f.stat().st_size
                f.unlink()
                deleted_raw_size += sz
                deleted_raw_count += 1
            except Exception as e:
                print(f"Hata ({f.name}): {e}")
print(f"0.2 - Silinen ham veri ({deleted_raw_count} dosya): {deleted_raw_size / (2**20):.1f} MB")

# 0.3 Eski Checkpoint Temizliği (checkpoints/)
ckpt_dir = Path("checkpoints")
KEEP = {"small_lm_best.pt", "small_lm_final.pt", "mergen_sft.pt", "tokenizer_bpe.json"}
deleted_ckpt_size = 0
deleted_ckpt_count = 0
if ckpt_dir.exists():
    for f in list(ckpt_dir.glob("*.pt")):
        if f.name not in KEEP:
            try:
                sz = f.stat().st_size
                f.unlink()
                deleted_ckpt_size += sz
                deleted_ckpt_count += 1
            except Exception as e:
                print(f"Hata ({f.name}): {e}")
print(f"0.3 - Silinen checkpoint ({deleted_ckpt_count} dosya): {deleted_ckpt_size / (2**20):.1f} MB")

# 0.4 Eski İşlenmiş Veri Temizliği (data/processed/)
proc_dir = Path("data/processed")
deleted_proc_size = 0
deleted_proc_count = 0
if proc_dir.exists():
    for pattern in ["bpe_train*.bin", "bpe_val*.bin", "*.bak", "charset_v1*"]:
        for f in list(proc_dir.glob(pattern)):
            try:
                sz = f.stat().st_size
                f.unlink()
                deleted_proc_size += sz
                deleted_proc_count += 1
                print(f"Silindi: {f.name} ({sz / (2**20):.1f} MB)")
            except Exception as e:
                print(f"Hata ({f.name}): {e}")
print(f"0.4 - Silinen işlenmiş veri ({deleted_proc_count} dosya): {deleted_proc_size / (2**20):.1f} MB")

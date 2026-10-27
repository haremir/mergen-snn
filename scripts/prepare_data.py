"""Mergen-SNN — Stage 2 Veri Hazırlama Hattı (Char-level Dil Modeli).

data/raw/cs_corpus altındaki metinleri toplar, karakter setini (charset.json)
çıkarır, metni uint16 token dizisine dönüştürür ve train.bin / val.bin olarak kaydeder.
"""

import json
from pathlib import Path
import shutil
import sys
import time
from collections import Counter
from typing import Dict, Tuple

import numpy as np

from mergen.utils.logger import get_logger

# Windows konsolunda Unicode çıktı uyumluluğu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = get_logger("mergen.prepare_data")

CORPUS_DIR = Path("data/raw/cs_corpus")
OUT_DIR = Path("data/processed")
TOKENIZER_DIR = Path("data/raw/tokenizers")
SKIP_FILES = {"LICENSE_NOTE.txt", ".gitkeep", "_low_quality.txt"}


def collect_texts(corpus_dir: Path) -> str:
    """data/raw/cs_corpus/ altındaki tüm *.txt dosyalarını toplar.

    Atlanan dosyalar: LICENSE_NOTE.txt, .gitkeep, _low_quality.txt ve < 100 byte dosyalar.
    Dosyalar arasına "\n\n---\n\n" ayırıcı eklenir.
    """
    logger.info(f"Metin dosyaları taranıyor: {corpus_dir}")
    txt_files = sorted(corpus_dir.rglob("*.txt"))
    collected = []
    skipped_count = 0

    for fpath in txt_files:
        if fpath.name in SKIP_FILES:
            skipped_count += 1
            continue
        try:
            size = fpath.stat().st_size
            if size < 100:
                skipped_count += 1
                continue
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            if len(text.strip()) >= 50:
                collected.append(text)
            else:
                skipped_count += 1
        except Exception as e:
            logger.warning(f"Dosya okuma hatası ({fpath}): {e}")
            skipped_count += 1

    logger.info(f"Toplanan metin dosyası sayısı: {len(collected)} (Atlanan: {skipped_count})")
    merged_text = "\n\n---\n\n".join(collected)
    return merged_text


def build_charset(text: str, min_freq: int = 5) -> dict:
    """Karakter frekanslarını sayar ve min_freq eşiğini geçen sıralı karakter kümesini oluşturur."""
    logger.info(f"Karakter seti inşa ediliyor (min_freq={min_freq})...")
    counts = Counter(text)
    filtered_chars = [ch for ch, cnt in counts.items() if cnt >= min_freq]
    chars = sorted(filtered_chars)
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    freq = {ch: counts[ch] for ch in chars}

    logger.info(f"Vocab size: {vocab_size} (Frekansı < {min_freq} olan atılan nadir karakter sayısı: {len(counts) - vocab_size})")
    return {
        "chars": chars,
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
        "freq": freq,
    }


def split_and_save(text: str, charset: dict, out_dir: Path, train_ratio: float = 0.9):
    """Metni karakter indekslerine çevirir, %90 train / %10 val olarak böler ve diske kaydeder."""
    stoi = charset["stoi"]
    vocab_size = charset["vocab_size"]
    out_dir.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    # Veri tipi güvenliği: vocab < 65536 için uint16
    if vocab_size < 65536:
        dtype = np.uint16
    else:
        logger.warning(f"Vocab size ({vocab_size}) >= 65536, uint32 kullanılıyor.")
        dtype = np.uint32

    logger.info("Metin token indekslerine kodlanıyor...")
    encoded = [stoi[ch] for ch in text if ch in stoi]
    tokens_arr = np.array(encoded, dtype=dtype)
    total_tokens = len(tokens_arr)

    split_idx = int(total_tokens * train_ratio)
    train_arr = tokens_arr[:split_idx]
    val_arr = tokens_arr[split_idx:]

    print(f"Train tokens: {len(train_arr):,} | Val tokens: {len(val_arr):,} | Toplam: {total_tokens:,}")

    # Dosyaları yaz (np.save ile doğrudan .bin dosyasına yazma)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    charset_path = out_dir / "charset.json"
    tok_charset_path = TOKENIZER_DIR / "charset.json"

    with open(train_path, "wb") as f:
        np.save(f, train_arr)
    with open(val_path, "wb") as f:
        np.save(f, val_arr)

    with open(charset_path, "w", encoding="utf-8") as f:
        json.dump(charset, f, ensure_ascii=False, indent=2)

    shutil.copy2(charset_path, tok_charset_path)

    print(f"charset.json, train.bin, val.bin yazıldı -> {out_dir}")
    print(f"charset.json kopyalandı -> {tok_charset_path}")


def main():
    start_time = time.time()
    logger.info("==========================================")
    logger.info("Mergen-SNN — Stage 2 Veri Hazırlama Hattı")
    logger.info("==========================================")

    # 1. Metinleri topla
    merged_text = collect_texts(CORPUS_DIR)
    print(f"Toplam karakter: {len(merged_text):,}")

    # 2. Charset oluştur
    charset = build_charset(merged_text, min_freq=5)
    vocab_size = charset["vocab_size"]
    print(f"Vocab size: {vocab_size}")

    # En sık 20 karakter
    top20 = Counter(charset["freq"]).most_common(20)
    print("\nEn sık 20 karakter:")
    for ch, cnt in top20:
        print(f"  {repr(ch):<10}: {cnt:,}")
    print()

    # 3. Böl ve diske kaydet
    split_and_save(merged_text, charset, OUT_DIR, train_ratio=0.9)

    elapsed = time.time() - start_time
    logger.info(f"Veri hazırlığı tamamlandı ({elapsed:.1f} saniye).")


if __name__ == "__main__":
    main()

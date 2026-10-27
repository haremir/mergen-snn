"""Mergen-SNN — Stage 2 Veri Hazırlama Hattı (Char-level Dil Modeli).

data/raw/cs_corpus altındaki metinleri toplar, kontrol karakterlerini temizler,
REQUIRED_CHARS (ASCII + Türkçe + temel noktalama) ve min_freq>=50 karakterlerle
kompakt bir charset oluşturur, <unk> (id=0) mekanizmasını entegre eder, metni
uint16 token dizisine dönüştürür ve train.bin / val.bin olarak kaydeder.
"""

from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, Tuple
import unicodedata

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

# Frekansa bakılmaksızın her zaman charset'te yer alacak zorunlu karakterler
REQUIRED_CHARS = (
    # ASCII printable (95 karakter, kod 32-126)
    "".join(chr(i) for i in range(32, 127))
    # Türkçe (12 karakter)
    + "çÇğĞıİöÖşŞüÜ"
    # Temel noktalama (ASCII dışı ama yaygın)
    + "‘’“”–—…«»"
)

LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u0152": "OE",
    "\u0153": "oe",
    "\u00c6": "AE",
    "\u00e6": "ae",
}


def clean_text(text: str) -> str:
    """Ligatürleri normalize et, non-breaking space dönüştür ve C0/C1 kontrolleri at."""
    # 1) Ligatürleri normalize et
    for lig, repl in LIGATURE_MAP.items():
        text = text.replace(lig, repl)

    # 2) Non-breaking space -> normal boşluk
    text = text.replace("\xa0", " ")

    # 3) Kontrol karakterlerini at (C0: 0-31, C1: 128-159, DEL: 127)
    allowed = {"\t", "\n", "\r"}
    cleaned = []
    for ch in text:
        cp = ord(ch)
        if ch in allowed:
            cleaned.append(ch)
        elif cp < 32 or cp == 127:  # C0 + DEL
            continue
        elif 128 <= cp <= 159:  # C1
            continue
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def collect_texts(corpus_dir: Path) -> Tuple[str, int]:
    """data/raw/cs_corpus/ altındaki tüm geçerli *.txt dosyalarını toplar ve temizler.

    Atlanan dosyalar: LICENSE_NOTE.txt, .gitkeep, _low_quality.txt ve temizleme sonrası < 100 karakter dosyalar.
    Dosyalar arasına "\n\n---\n\n" ayırıcı eklenir.
    Dönen: (merged_text, removed_count)
    """
    logger.info(f"Metin dosyaları taranıyor: {corpus_dir}")
    txt_files = sorted(corpus_dir.rglob("*.txt"))
    collected = []
    skipped_count = 0
    raw_chars_total = 0
    cleaned_chars_total = 0

    for fpath in txt_files:
        if fpath.name in SKIP_FILES:
            skipped_count += 1
            continue
        try:
            size = fpath.stat().st_size
            if size < 100:
                skipped_count += 1
                continue
            content = fpath.read_text(encoding="utf-8", errors="ignore")
            raw_chars_total += len(content)
            content = clean_text(content)
            if len(content) < 100:
                skipped_count += 1
                continue
            cleaned_chars_total += len(content)
            collected.append(content)
        except Exception as e:
            logger.warning(f"Dosya okuma hatası ({fpath}): {e}")
            skipped_count += 1

    removed_count = raw_chars_total - cleaned_chars_total
    merged_text = "\n\n---\n\n".join(collected)
    logger.info(f"Toplanan metin dosyası sayısı: {len(collected)} (Atlanan: {skipped_count})")
    return merged_text, removed_count


def build_charset(text: str, min_freq: int = 50) -> dict:
    """Karakter frekanslarını sayar; REQUIRED_CHARS ve min_freq eşiğini geçen karakterlerle charset oluşturur.

    <unk> tokeni id=0 olarak başa eklenir.
    """
    logger.info(f"Karakter seti inşa ediliyor (min_freq={min_freq})...")
    counter = Counter(text)
    chars = set()

    # 1. Frekansa bakılmaksızın zorunlu karakterleri ekle
    for ch in REQUIRED_CHARS:
        chars.add(ch)

    # 2. Frekansı min_freq ve üzeri olan karakterleri ekle (ör. \n, \t)
    for ch, count in counter.items():
        if count >= min_freq:
            chars.add(ch)

    # 3. Sırala ve <unk> başa ekle
    sorted_chars = sorted(chars)
    final_chars = ["<unk>"] + sorted_chars

    # 4. stoi ve itos haritaları
    stoi = {"<unk>": 0}
    for i, ch in enumerate(sorted_chars):
        stoi[ch] = i + 1

    itos = {i: ch for ch, i in stoi.items()}

    logger.info(
        f"Vocab size: {len(final_chars)} (Zorunlu karakter: {len(REQUIRED_CHARS)}, "
        f"min_freq>={min_freq} eklenen: {len(sorted_chars) - len(REQUIRED_CHARS)}, <unk>: 1)"
    )
    return {
        "chars": final_chars,
        "stoi": stoi,
        "itos": itos,
        "vocab_size": len(final_chars),
        "freq": dict(counter),
    }


def split_and_save(text: str, charset: dict, out_dir: Path, train_ratio: float = 0.9):
    """Metni karakter indekslerine çevirir (<unk> eşlemesiyle), %90 train / %10 val olarak böler ve diske kaydeder."""
    stoi = charset["stoi"]
    vocab_size = charset["vocab_size"]
    unk_id = stoi.get("<unk>", 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    # Veri tipi güvenliği: vocab < 65536 için uint16
    if vocab_size < 65536:
        dtype = np.uint16
    else:
        logger.warning(f"Vocab size ({vocab_size}) >= 65536, uint32 kullanılıyor.")
        dtype = np.uint32

    logger.info("Metin token indekslerine kodlanıyor (<unk> fallback ile)...")
    unk_count = 0
    encoded = []
    for ch in text:
        if ch in stoi:
            encoded.append(stoi[ch])
        else:
            encoded.append(unk_id)
            unk_count += 1

    tokens_arr = np.array(encoded, dtype=dtype)
    total_tokens = len(tokens_arr)
    unk_ratio = (unk_count / total_tokens) * 100 if total_tokens > 0 else 0.0

    print(f"<unk> token sayısı: {unk_count:,} / {total_tokens:,} (oran: %{unk_ratio:.4f})")

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

    # 1. Metinleri topla ve kontrol karakterlerini temizle
    merged_text, removed_count = collect_texts(CORPUS_DIR)
    print(f"Atılan kontrol karakteri sayısı: {removed_count:,}")
    print(f"Temizlik sonrası toplam karakter: {len(merged_text):,}")

    # 2. Charset oluştur (min_freq=50, REQUIRED_CHARS garantili, <unk>=0)
    charset = build_charset(merged_text, min_freq=50)
    vocab_size = charset["vocab_size"]
    print(f"Vocab size (temizlik sonrası): {vocab_size}")

    print("\nCharset ilk 20 karakter (id 0-19):")
    for i in range(min(20, vocab_size)):
        ch = charset["chars"][i]
        print(f"  id {i:>2}: {repr(ch)}")

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

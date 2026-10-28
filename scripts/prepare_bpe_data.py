"""Mergen-SNN — Stage 3: BPE ile Veri Kümesi Hazırlama (bpe_train.bin / bpe_val.bin)."""

from pathlib import Path
import sys
import time
from typing import List

import numpy as np
from tokenizers import Tokenizer

from mergen.utils.logger import get_logger

# Windows konsolunda Unicode çıktı uyumluluğu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = get_logger("mergen.prepare_bpe_data")

CORPUS_DIR = Path("data/raw/cs_corpus")
OUT_DIR = Path("data/processed")
TOKENIZER_PATH = OUT_DIR / "tokenizer_bpe.json"
SKIP_FILES = {"LICENSE_NOTE.txt", ".gitkeep", "_low_quality.txt"}

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
    """Ligatürleri normalize et, non-breaking space dönüştür ve C0/C1 kontrolleri ayıkla."""
    for lig, repl in LIGATURE_MAP.items():
        text = text.replace(lig, repl)
    text = text.replace("\xa0", " ")
    allowed = {"\t", "\n", "\r"}
    cleaned = []
    for ch in text:
        cp = ord(ch)
        if ch in allowed:
            cleaned.append(ch)
        elif cp < 32 or cp == 127 or (128 <= cp <= 159):
            continue
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def collect_texts(corpus_dir: Path) -> List[str]:
    """CS korpusundaki geçerli metinleri temizlenmiş olarak döndürür."""
    logger.info(f"Metin dosyaları taranıyor: {corpus_dir}")
    txt_files = sorted(corpus_dir.rglob("*.txt"))
    valid_files = [f for f in txt_files if f.name not in SKIP_FILES]
    collected = []
    skipped = 0

    for fpath in valid_files:
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
            content = clean_text(content)
            if len(content.strip()) >= 50:
                collected.append(content)
            else:
                skipped += 1
        except Exception as e:
            logger.warning(f"Dosya okunamadı ({fpath}): {e}")
            skipped += 1

    logger.info(f"Toplanan metin sayısı: {len(collected)} (Atlanan: {skipped})")
    return collected


def main():
    logger.info("==========================================")
    logger.info("Mergen-SNN — Stage 3: BPE Veri Hazırlama")
    logger.info("==========================================")

    if not TOKENIZER_PATH.exists():
        logger.error(f"Tokenizer bulunamadı: {TOKENIZER_PATH}")
        logger.error("Lütfen önce 'scripts/train_tokenizer.py' scriptini çalıştırın.")
        sys.exit(1)

    # 1. Tokenizer'ı yükle
    logger.info(f"Tokenizer yükleniyor: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Yüklenen Tokenizer Vocab Size: {vocab_size}")

    # 2. Metinleri topla
    t0 = time.time()
    texts = collect_texts(CORPUS_DIR)
    merged_text = "\n\n---\n\n".join(texts)
    logger.info(f"Birleştirilen toplam karakter: {len(merged_text):,}")

    # 3. Metni BPE ile kodla
    logger.info("Metin BPE ile token dizisine kodlanıyor...")
    t_enc_start = time.time()

    # Bellek ve işlem verimliliği için parçalı encode
    chunk_size = 500_000
    all_token_ids: List[int] = []
    for i in range(0, len(merged_text), chunk_size):
        sub_text = merged_text[i : i + chunk_size]
        encoding = tokenizer.encode(sub_text)
        all_token_ids.extend(encoding.ids)

    enc_duration = time.time() - t_enc_start
    total_tokens = len(all_token_ids)
    logger.info(f"Kodlama tamamlandı ({enc_duration:.2f}s). Toplam token: {total_tokens:,}")

    # 4. uint16 tipine çevir ve train/val olarak böl
    if vocab_size < 65536:
        dtype = np.uint16
    else:
        logger.warning(f"Vocab ({vocab_size}) >= 65536, uint32 kullanılıyor.")
        dtype = np.uint32

    tokens_arr = np.array(all_token_ids, dtype=dtype)

    train_ratio = 0.90
    split_idx = int(total_tokens * train_ratio)
    train_tokens = tokens_arr[:split_idx]
    val_tokens = tokens_arr[split_idx:]

    # 5. Dosyaları kaydet
    train_out_path = OUT_DIR / "bpe_train.bin"
    val_out_path = OUT_DIR / "bpe_val.bin"

    with open(train_out_path, "wb") as f:
        np.save(f, train_tokens)

    with open(val_out_path, "wb") as f:
        np.save(f, val_tokens)

    train_size_mb = train_out_path.stat().st_size / (1024 * 1024)
    val_size_mb = val_out_path.stat().st_size / (1024 * 1024)

    total_time = time.time() - t0

    print("\n" + "=" * 60)
    print("BPE VERİ HAZIRLAMA RAPORU")
    print("=" * 60)
    print(f"Toplam Token Sayısı : {total_tokens:,}")
    print(f"Train Tokens (%90)  : {len(train_tokens):,} ({train_size_mb:.2f} MB)")
    print(f"Val Tokens (%10)    : {len(val_tokens):,} ({val_size_mb:.2f} MB)")
    print(f"Kayıt Formatı       : {dtype.__name__} (NumPy binary array)")
    print(f"Çıktı Dosyaları     : {train_out_path.name}, {val_out_path.name}")
    print(f"Toplam İşlem Süresi : {total_time:.2f} saniye")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

"""Mergen-SNN — Stage 3: Byte-Level BPE Tokenizer Eğitimi."""

from pathlib import Path
import sys
import time

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from mergen.utils.logger import get_logger

# Windows konsolunda Unicode çıktı uyumluluğu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = get_logger("mergen.train_tokenizer")

CORPUS_DIR = Path("data/raw/cs_corpus")
OUT_DIR = Path("data/processed")
TOKENIZER_DIR = Path("data/raw/tokenizers")
SKIP_FILES = {"LICENSE_NOTE.txt", ".gitkeep", "_low_quality.txt"}

# Özel tokenler (Stage 3 ve ilerisi için LLM / FIM / Tool tokenleri)
SPECIAL_TOKENS = [
    "<unk>",
    "<pad>",
    "<endoftext>",
    "<system>",
    "<user>",
    "<assistant>",
    "<tool_call>",
    "<tool_result>",
    "<fim_prefix>",
    "<fim_middle>",
    "<fim_suffix>",
]

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


def load_corpus_texts(corpus_dir: Path):
    """CS korpusundaki metinleri temizleyerek liste olarak döndürür."""
    logger.info(f"Metin dosyaları taranıyor: {corpus_dir}")
    txt_files = sorted(corpus_dir.rglob("*.txt"))
    valid_files = [f for f in txt_files if f.name not in SKIP_FILES]
    logger.info(f"Toplam geçerli dosya: {len(valid_files)}")

    texts = []
    total_chars = 0
    for fpath in valid_files:
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
            content = clean_text(content)
            if len(content.strip()) >= 50:
                texts.append(content)
                total_chars += len(content)
        except Exception as e:
            logger.warning(f"Dosya okunamadı ({fpath}): {e}")

    logger.info(f"Toplanan metin sayısı: {len(texts)} (Toplam karakter: {total_chars:,})")
    return texts, total_chars


def main():
    logger.info("==========================================")
    logger.info("Mergen-SNN — Stage 3: Byte-Level BPE Tokenizer Eğitimi")
    logger.info("==========================================")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Metinleri topla
    t_start = time.time()
    texts, total_chars = load_corpus_texts(CORPUS_DIR)

    # 2. ByteLevel BPE Tokenizer modeli ve önişleyici kur
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=8192,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # 3. Tokenizer'ı eğit
    logger.info("Tokenizer eğitimi başlatılıyor (vocab_size=8192, min_frequency=2)...")
    t_train_start = time.time()
    tokenizer.train_from_iterator(texts, trainer)
    train_duration = time.time() - t_train_start

    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Eğitim tamamlandı! Süre: {train_duration:.2f}s | Vocab size: {vocab_size}")

    # 4. Tokenizer'ı kaydet
    out_path = OUT_DIR / "tokenizer_bpe.json"
    tok_dir_path = TOKENIZER_DIR / "tokenizer_bpe.json"
    tokenizer.save(str(out_path))
    tokenizer.save(str(tok_dir_path))
    logger.info(f"Tokenizer kaydedildi: {out_path} ve {tok_dir_path}")

    # 5. İstatistikleri ve sıkıştırma oranını hesapla
    logger.info("Külliyat üzerindeki toplam token sayısı ve sıkıştırma oranı hesaplanıyor...")
    total_tokens = 0
    # Batch encoding ile hızlandır
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encodings = tokenizer.encode_batch(batch)
        for enc in encodings:
            total_tokens += len(enc.ids)

    compression_ratio = total_chars / max(total_tokens, 1)

    print("\n" + "=" * 60)
    print("TOKENIZER EĞİTİM VE SIKIŞTIRMA ÖZETİ")
    print("=" * 60)
    print(f"Eğitim Süresi         : {train_duration:.2f} saniye")
    print(f"Toplam Vocab Boyutu   : {vocab_size}")
    print(f"Özel Token Sayısı     : {len(SPECIAL_TOKENS)}")
    print(f"Toplam Karakter Sayısı: {total_chars:,}")
    print(f"Toplam Token Sayısı   : {total_tokens:,}")
    print(f"Sıkıştırma Oranı      : {compression_ratio:.2f} karakter / token")
    print(f"Tasarruf Oranı        : %{(1 - (total_tokens / total_chars)) * 100:.2f}")
    print(f"Hedef Dosya           : {out_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

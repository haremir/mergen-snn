"""Mergen-SNN — Massive BPE Tokenizer Eğitimi (vocab_size=16384).

Streaming kaynaklardan bellek dostu biçimde BPE sözlüğü üretir.
"""

from pathlib import Path
import shutil
import sys
import time

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from mergen.data.streaming_sources import stream_all_sources
from mergen.utils.logger import get_logger

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = get_logger("mergen.train_tokenizer_massive")

OUT_DIR = Path("data/processed")
OLD_TOKENIZER = OUT_DIR / "tokenizer_bpe.json"
BAK_TOKENIZER = OUT_DIR / "tokenizer_bpe_v1.json.bak"
NEW_TOKENIZER = OUT_DIR / "tokenizer_bpe_v2.json"

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


def main():
    logger.info("==========================================")
    logger.info("Mergen-SNN — Massive Byte-Level BPE Tokenizer (16K Vocab)")
    logger.info("==========================================")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Eski tokenizer'ı yedekle
    if OLD_TOKENIZER.exists():
        shutil.copy2(OLD_TOKENIZER, BAK_TOKENIZER)
        logger.info(f"Eski tokenizer yedeklendi: {OLD_TOKENIZER} -> {BAK_TOKENIZER}")

    # 2. Tokenizer model ve ön-işleyici kurulumu
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=16384,
        min_frequency=3,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # 3. Tokenizer eğitimi için tüm alanları temsil eden dengeli streaming örneklemi (toplam ~150 MB)
    sample_quotas = {
        "Mevcut Local TXT": 40 * 1024 * 1024,
        "CulturaX / TR": 30 * 1024 * 1024,
        "The Stack / Code": 40 * 1024 * 1024,
        "S2ORC / ArXiv": 20 * 1024 * 1024,
        "FineWeb": 20 * 1024 * 1024,
    }

    logger.info("Streaming kaynaklardan temsilci metin akışı alınıyor...")
    collected_texts = []
    total_bytes = 0

    for src, text in stream_all_sources(quotas=sample_quotas):
        collected_texts.append(text)
        total_bytes += len(text.encode("utf-8"))
        if len(collected_texts) % 2000 == 0:
            logger.info(f"Toplanan temsilci metin: {len(collected_texts):,} döküman ({total_bytes / (1024*1024):.1f} MB)")

    logger.info(f"Eğitim veri kümesi hazır: {len(collected_texts):,} döküman ({total_bytes / (1024*1024):.2f} MB)")
    logger.info("BPE Trainer çalıştırılıyor (vocab_size=16384, min_frequency=3)...")

    t_start = time.time()
    tokenizer.train_from_iterator(collected_texts, trainer=trainer)
    train_sec = time.time() - t_start

    # 4. Kaydet
    tokenizer.save(str(NEW_TOKENIZER))
    logger.info(f"Yeni tokenizer kaydedildi: {NEW_TOKENIZER} (Süre: {train_sec:.2f}s)")

    # 5. Doğrulama testleri
    test_sentences = [
        "Python'da liste ve sözlük veri yapıları nasıl kullanılır?",
        "def train(model, dataloader, optimizer): return loss",
        "The transformer architecture relies on multi-head self-attention mechanisms.",
        "Ayrık matematik, graf teorisi ve algoritma analizi bilgisayar biliminin temelidir.",
        "<system>Sen Mergen'sin.</system><user>Merhaba</user><assistant>Selam!</assistant>",
    ]

    print("\n--- Tokenizer v2 Doğrulama Testleri ---")
    for s in test_sentences:
        enc = tokenizer.encode(s)
        dec = tokenizer.decode(enc.ids)
        print(f"Metin: {s}")
        print(f"Token Sayısı: {len(enc.ids)} | Sıkıştırma: {len(s) / max(len(enc.ids), 1):.2f} char/tok")
        print(f"Decode Doğruluk: {'TAM UYUŞUYOR' if dec == s else 'FARKLI'}\n")

    vocab = tokenizer.get_vocab()
    print(f"Nihai Vocab Boyutu: {len(vocab):,}")


if __name__ == "__main__":
    main()

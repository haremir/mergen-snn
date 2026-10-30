"""Mergen-SNN — Massive Binary Veri Üretimi (Streaming, Disk Dostu).

Ham veriyi diske kaydetmeden doğrudan streaming kaynaklardan okur,
yeni 16K BPE tokenizer ile encode eder ve doğrudan uint16 binary dosyalara (append) yazar.
%98 Train (bpe_train_v2.bin), %2 Val (bpe_val_v2.bin).
"""

import gc
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict

import numpy as np
from tokenizers import Tokenizer

from mergen.data.streaming_sources import stream_all_sources
from mergen.utils.logger import get_logger

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = get_logger("mergen.prepare_bpe_data_massive")

OUT_DIR = Path("data/processed")
TOKENIZER_PATH = OUT_DIR / "tokenizer_bpe_v2.json"
TRAIN_BIN = OUT_DIR / "bpe_train_v2.bin"
VAL_BIN = OUT_DIR / "bpe_val_v2.bin"


def main():
    logger.info("==========================================")
    logger.info("Mergen-SNN — Massive Streaming Binary Veri Hazırlama")
    logger.info("==========================================")

    if not TOKENIZER_PATH.exists():
        logger.error(f"Tokenizer bulunamadı: {TOKENIZER_PATH}. Önce 'scripts/train_tokenizer_massive.py' çalıştırılmalı!")
        sys.exit(1)

    logger.info(f"Tokenizer yükleniyor: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Yüklenen Tokenizer Vocab Boyutu: {vocab_size:,}")

    eot_id = tokenizer.token_to_id("<endoftext>")
    if eot_id is None:
        eot_id = tokenizer.token_to_id("<unk>") or 0
    logger.info(f"Document separator (<endoftext>) token id: {eot_id}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Hedef Kotalar (Toplam ~5 GB metin)
    quotas = {
        "Mevcut Local TXT": 500 * 1024 * 1024,
        "CulturaX / TR": 1000 * 1024 * 1024,
        "The Stack / Code": 1500 * 1024 * 1024,
        "S2ORC / ArXiv": 1000 * 1024 * 1024,
        "FineWeb": 1000 * 1024 * 1024,
    }

    # İstatistik takipçileri
    source_stats = {k: {"bytes": 0, "docs": 0, "tokens": 0} for k in quotas}
    total_train_tokens = 0
    total_val_tokens = 0
    total_docs = 0
    total_bytes = 0

    t_start = time.time()
    t_last_log = t_start

    # Binary dosyaları sıfırdan aç
    with open(TRAIN_BIN, "wb") as f_train, open(VAL_BIN, "wb") as f_val:
        train_buffer = []
        val_buffer = []
        buffer_token_count = 0
        BUFFER_FLUSH_TOKENS = 500_000

        batch_texts = []
        batch_srcs = []
        BATCH_SIZE = 128

        random.seed(42)

        def process_batch(texts, srcs):
            nonlocal total_train_tokens, total_val_tokens, buffer_token_count, total_bytes, total_docs
            if not texts:
                return
            encs = tokenizer.encode_batch(texts)
            for src, text, enc in zip(srcs, texts, encs):
                ids = enc.ids
                if not ids:
                    continue
                doc_tokens = ids + [eot_id]
                doc_len = len(doc_tokens)
                b_len = len(text.encode("utf-8"))

                source_stats[src]["bytes"] += b_len
                source_stats[src]["docs"] += 1
                source_stats[src]["tokens"] += doc_len
                total_bytes += b_len
                total_docs += 1

                if random.random() < 0.98:
                    train_buffer.extend(doc_tokens)
                    total_train_tokens += doc_len
                else:
                    val_buffer.extend(doc_tokens)
                    total_val_tokens += doc_len
                buffer_token_count += doc_len

        for src, text in stream_all_sources(quotas=quotas):
            batch_texts.append(text)
            batch_srcs.append(src)

            if len(batch_texts) >= BATCH_SIZE:
                process_batch(batch_texts, batch_srcs)
                batch_texts.clear()
                batch_srcs.clear()

            # Buffer dolunca diske yaz
            if buffer_token_count >= BUFFER_FLUSH_TOKENS:
                if train_buffer:
                    np.array(train_buffer, dtype=np.uint16).tofile(f_train)
                    train_buffer.clear()
                if val_buffer:
                    np.array(val_buffer, dtype=np.uint16).tofile(f_val)
                    val_buffer.clear()
                buffer_token_count = 0
                f_train.flush()
                f_val.flush()

            # İlerleme Loglama (her 1000 dökümanda bir)
            if total_docs % 1000 == 0 and total_docs > 0:
                now = time.time()
                elapsed = now - t_start
                total_tok = total_train_tokens + total_val_tokens
                tok_per_sec = total_tok / max(elapsed, 0.1)
                mb_text = total_bytes / (1024 * 1024)
                mb_bin = (total_tok * 2) / (1024 * 1024)

                logger.info(
                    f"[{src}] {total_docs:,} doküman | Metin: {mb_text:.1f} MB | "
                    f"Token: {total_tok:,} (~{total_tok/1e6:.1f}M) | Binary: {mb_bin:.1f} MB | "
                    f"Hız: {tok_per_sec:,.0f} tok/s"
                )
                gc.collect()

        # Son batch'i işle
        if batch_texts:
            process_batch(batch_texts, batch_srcs)
            batch_texts.clear()
            batch_srcs.clear()

        # Kalan buffer'ı diske boşalt
        if train_buffer:
            arr_train = np.array(train_buffer, dtype=np.uint16)
            arr_train.tofile(f_train)
            train_buffer.clear()
        if val_buffer:
            arr_val = np.array(val_buffer, dtype=np.uint16)
            arr_val.tofile(f_val)
            val_buffer.clear()
        f_train.flush()
        f_val.flush()

    total_time = time.time() - t_start
    total_tokens = total_train_tokens + total_val_tokens
    train_size_mb = TRAIN_BIN.stat().st_size / (1024 * 1024)
    val_size_mb = VAL_BIN.stat().st_size / (1024 * 1024)

    logger.info("\n" + "=" * 60)
    logger.info("MASSIVE STREAMING BINARY HAZIRLAMA TAMAMLANDI!")
    logger.info("=" * 60)
    for src, st in source_stats.items():
        src_mb = st["bytes"] / (1024 * 1024)
        src_tok = st["tokens"]
        logger.info(f"  {src:<22}: {st['docs']:>8,} doküman | {src_mb:>8.2f} MB | {src_tok:>10,} token")

    logger.info("-" * 60)
    logger.info(f"Toplam Metin Boyutu : {total_bytes / (1024*1024*1024):.2f} GB ({total_bytes / (1024*1024):.1f} MB)")
    logger.info(f"Toplam Doküman Sayısı: {total_docs:,}")
    logger.info(f"Train Tokens         : {total_train_tokens:,} ({train_size_mb:.2f} MB -> {TRAIN_BIN})")
    logger.info(f"Val Tokens           : {total_val_tokens:,} ({val_size_mb:.2f} MB -> {VAL_BIN})")
    logger.info(f"Toplam Token Sayısı  : {total_tokens:,} (~{total_tokens/1e6:.2f}M token)")
    logger.info(f"Toplam Geçen Süre    : {total_time:.1f} saniye ({total_time/60:.1f} dakika)")
    logger.info(f"Ortalama Verim       : {total_tokens / max(total_time, 0.1):,.0f} token/saniye")


if __name__ == "__main__":
    main()

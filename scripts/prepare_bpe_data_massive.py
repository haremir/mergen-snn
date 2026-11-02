"""Mergen-SNN — Massive Binary Veri Üretimi v3 (Dengeli Karışım, Disk Dostu).

Ham veriyi diske kaydetmeden doğrudan streaming kaynaklardan okur,
yeni 16K BPE tokenizer ile encode eder ve doğrudan uint16 binary dosyalara (append) yazar.
Yeni Oranlar:
- %55 Kod (The Stack / codeparrot, ~594M token)
- %15 Türkçe Wikipedia (CulturaX / wikimedia, ~162M token)
- %15 FineWeb (~162M token)
- %10 ML-ArXiv / S2ORC (~108M token)
- %5 Mevcut Korpus (.txt, ~54M token)
Toplam: 1.080.000.000 Token (~1.08B), %98 Train (bpe_train_v3.bin), %2 Val (bpe_val_v3.bin).
"""

import gc
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

import numpy as np
from tokenizers import Tokenizer

from mergen.data.streaming_sources import (
    stream_all_sources,
    DEFAULT_BYTE_QUOTAS,
    DEFAULT_TOKEN_QUOTAS,
)
from mergen.utils.logger import get_logger

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = get_logger("mergen.prepare_bpe_data_massive")

OUT_DIR = Path("data/processed")
TOKENIZER_PATH = OUT_DIR / "tokenizer_bpe_v2.json"
TRAIN_BIN = OUT_DIR / "bpe_train_v3.bin"
VAL_BIN = OUT_DIR / "bpe_val_v3.bin"

# 1.08B Token Hedef Kotaları
TARGET_TOKEN_QUOTAS = DEFAULT_TOKEN_QUOTAS.copy()
TOTAL_TARGET_TOKENS = sum(TARGET_TOKEN_QUOTAS.values())  # 1,080,000,000


def main():
    logger.info("==========================================")
    logger.info("Mergen-SNN — Massive Binary Veri Hazırlama v3 (Dengeli Karışım)")
    logger.info("==========================================")
    logger.info("Hedef Oranlar:")
    for src, tok in TARGET_TOKEN_QUOTAS.items():
        pct = (tok / TOTAL_TARGET_TOKENS) * 100
        logger.info(f" - {src:<20}: {tok:,} token (%{pct:.1f})")
    logger.info(f"Toplam Hedef Token Sayısı: {TOTAL_TARGET_TOKENS:,} (~1.08B)")
    logger.info("==========================================")

    if not TOKENIZER_PATH.exists():
        logger.error(f"Tokenizer bulunamadı: {TOKENIZER_PATH}!")
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

    source_stats = {k: {"bytes": 0, "docs": 0, "tokens": 0} for k in TARGET_TOKEN_QUOTAS}
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

        def process_batch(texts, srcs):
            nonlocal total_train_tokens, total_val_tokens, buffer_token_count, total_bytes, total_docs
            if not texts:
                return
            encs = tokenizer.encode_batch(texts)
            for src, text, enc in zip(srcs, texts, encs):
                ids = enc.ids
                if not ids:
                    continue

                # Kaynağın kotasını kontrol et
                target_q = TARGET_TOKEN_QUOTAS.get(src, float("inf"))
                current_tokens = source_stats[src]["tokens"]
                if current_tokens >= target_q:
                    continue

                doc_tokens = ids + [eot_id]
                # Eğer kota aşılacaksa sadece kalan kadarını al
                if current_tokens + len(doc_tokens) > target_q:
                    allowed_len = max(target_q - current_tokens, 0)
                    doc_tokens = doc_tokens[:allowed_len]
                    if not doc_tokens:
                        continue

                doc_len = len(doc_tokens)
                b_len = len(text.encode("utf-8"))

                source_stats[src]["bytes"] += b_len
                source_stats[src]["docs"] += 1
                source_stats[src]["tokens"] += doc_len
                total_bytes += b_len
                total_docs += 1

                # %98 Train, %2 Val
                if random.random() < 0.98:
                    train_buffer.extend(doc_tokens)
                    total_train_tokens += doc_len
                else:
                    val_buffer.extend(doc_tokens)
                    total_val_tokens += doc_len
                buffer_token_count += doc_len

        for src, text in stream_all_sources(interleave=True):
            # Eğer tüm kotalar dolduysa sonlandır
            total_current = total_train_tokens + total_val_tokens
            if total_current >= TOTAL_TARGET_TOKENS:
                break

            # Kaynağın kotası dolmuşsa pas geç
            if source_stats.get(src, {}).get("tokens", 0) >= TARGET_TOKEN_QUOTAS.get(src, float("inf")):
                continue

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

            # İlerleme Loglama (her 2500 dökümanda bir)
            if total_docs % 2500 == 0 and total_docs > 0:
                now = time.time()
                elapsed = now - t_start
                total_tok = total_train_tokens + total_val_tokens
                tok_per_sec = total_tok / max(elapsed, 0.1)
                mb_text = total_bytes / (1024 * 1024)
                mb_bin = (total_tok * 2) / (1024 * 1024)
                pct = (total_tok / TOTAL_TARGET_TOKENS) * 100

                logger.info(
                    f"[{src}] {total_docs:,} doküman | Metin: {mb_text:.1f} MB | "
                    f"Token: {total_tok:,} / {TOTAL_TARGET_TOKENS:,} (%{pct:.1f}) | "
                    f"Binary: {mb_bin:.1f} MB | Hız: {tok_per_sec:,.0f} tok/s"
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
    train_size_mb = TRAIN_BIN.stat().st_size / (1024 * 1024) if TRAIN_BIN.exists() else 0
    val_size_mb = VAL_BIN.stat().st_size / (1024 * 1024) if VAL_BIN.exists() else 0

    logger.info("\n" + "=" * 60)
    logger.info("MASSIVE STREAMING BINARY HAZIRLAMA v3 TAMAMLANDI!")
    logger.info("=" * 60)
    for src, st in source_stats.items():
        src_mb = st["bytes"] / (1024 * 1024)
        src_tok = st["tokens"]
        target_tok = TARGET_TOKEN_QUOTAS.get(src, 1)
        logger.info(f"  {src:<22}: {st['docs']:>8,} doküman | {src_mb:>8.2f} MB | {src_tok:>11,} token (%{src_tok/total_tokens*100:.1f})")

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

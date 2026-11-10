"""Mergen-SNN — Massive Binary Veri Üretimi v3b (2B Token, Disk Dostu Streaming).

Ham veriyi diske kaydetmeden doğrudan streaming kaynaklardan okur,
yeni 16K BPE tokenizer ile encode eder ve doğrudan uint16 binary dosyalara (append) yazar.
Kotalar:
- %45 Kod (The Stack / codeparrot, ~920M token / 3.0 GB)
- %30 FineWeb (~610M token / 2.0 GB)
- %16 Türkçe Wikipedia (CulturaX / wikimedia, ~330M token / 1.0 GB)
- %7.5 ML-ArXiv / S2ORC (~150M token / 0.5 GB)
- %2 Mevcut Korpus (.txt, ~40M token / 0.12 GB)
Toplam: ~2.05 Milyar Token (~2B), %98 Train (bpe_train_v3b.bin), %2 Val (bpe_val_v3b.bin).
"""

import argparse
import gc
import os
from pathlib import Path
import random
import shutil
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
TRAIN_BIN = OUT_DIR / "bpe_train_v3b.bin"
VAL_BIN = OUT_DIR / "bpe_val_v3b.bin"

# 2.05B Token Hedef Kotaları
TARGET_TOKEN_QUOTAS = DEFAULT_TOKEN_QUOTAS.copy()
TOTAL_TARGET_TOKENS = sum(TARGET_TOKEN_QUOTAS.values())  # 2,050,000,000


def check_disk_space(required_gb: float = 15.0, target_path: Path = OUT_DIR) -> float:
    """Disk boş alanını kontrol et. 15 GB altındaysa durdur ve uyarı ver."""
    target_path.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(target_path)
    free_gb = free / (1024 ** 3)
    logger.info(f"Disk Alanı Kontrolü: {free_gb:.2f} GB boş alan mevcut (Gereken asgari: {required_gb:.1f} GB)")
    if free_gb < required_gb:
        logger.error(
            f"❌ KRİTİK HATA: Yetersiz disk alanı! Mevcut: {free_gb:.2f} GB < Gereken: {required_gb:.1f} GB. "
            "İşlem disk taşması riskini önlemek için durduruluyor!"
        )
        sys.exit(1)
    return free_gb


def main():
    parser = argparse.ArgumentParser(description="Massive BPE Data Preparation v3b (2B Tokens)")
    parser.add_argument("--train-out", type=str, default=str(TRAIN_BIN), help="Train binary output path")
    parser.add_argument("--val-out", type=str, default=str(VAL_BIN), help="Val binary output path")
    parser.add_argument("--tokenizer", type=str, default=str(TOKENIZER_PATH), help="Tokenizer path")
    parser.add_argument("--min-disk-gb", type=float, default=15.0, help="Minimum free disk space in GB required")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional max token limit for testing")
    parser.add_argument("--interleave", action="store_true", default=True, help="Interleave data sources homogeneously")
    args = parser.parse_args()

    train_out_path = Path(args.train_out)
    val_out_path = Path(args.val_out)
    tok_path = Path(args.tokenizer)

    # 1. Disk Kontrolü (İş 5)
    init_free_gb = check_disk_space(required_gb=args.min_disk_gb, target_path=train_out_path.parent)

    total_target = args.max_tokens or TOTAL_TARGET_TOKENS

    logger.info("==========================================")
    logger.info("Mergen-SNN — Massive Binary Veri Hazırlama v3b (2B Token Hedefi)")
    logger.info("==========================================")
    logger.info("Hedef Kotalar:")
    for src, tok in TARGET_TOKEN_QUOTAS.items():
        pct = (tok / TOTAL_TARGET_TOKENS) * 100
        gb = DEFAULT_BYTE_QUOTAS.get(src, 0) / (1024 ** 3)
        logger.info(f" - {src:<20}: {tok:,} token (%{pct:.1f}) | {gb:.2f} GB metin kotası")
    logger.info(f"Toplam Hedef Token Sayısı: {total_target:,} (~{total_target/1e9:.2f}B)")
    logger.info(f"Çıktı Dosyaları: {train_out_path} ve {val_out_path}")
    logger.info("==========================================")

    if not tok_path.exists():
        logger.error(f"Tokenizer bulunamadı: {tok_path}!")
        sys.exit(1)

    logger.info(f"Tokenizer yükleniyor: {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))
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

    # Binary dosyaları sıfırdan aç
    with open(train_out_path, "wb") as f_train, open(val_out_path, "wb") as f_val:
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

        for src, text in stream_all_sources(interleave=args.interleave):
            # Eğer toplam hedefe ulaşıldıysa sonlandır
            total_current = total_train_tokens + total_val_tokens
            if total_current >= total_target:
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
                pct = (total_tok / total_target) * 100

                logger.info(
                    f"[{src}] {total_docs:,} doküman | Metin: {mb_text:.1f} MB | "
                    f"Token: {total_tok:,} / {total_target:,} (%{pct:.1f}) | "
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
    train_size_mb = train_out_path.stat().st_size / (1024 * 1024) if train_out_path.exists() else 0
    val_size_mb = val_out_path.stat().st_size / (1024 * 1024) if val_out_path.exists() else 0
    final_free_gb = shutil.disk_usage(train_out_path.parent).free / (1024 ** 3)

    logger.info("\n" + "=" * 60)
    logger.info("MASSIVE STREAMING BINARY HAZIRLAMA v3b (2B TOKEN) TAMAMLANDI!")
    logger.info("=" * 60)
    for src, st in source_stats.items():
        src_mb = st["bytes"] / (1024 * 1024)
        src_tok = st["tokens"]
        logger.info(f"  {src:<22}: {st['docs']:>8,} doküman | {src_mb:>8.2f} MB | {src_tok:>11,} token (%{src_tok/max(total_tokens,1)*100:.1f})")

    logger.info("-" * 60)
    logger.info(f"Toplam Metin Boyutu : {total_bytes / (1024*1024*1024):.2f} GB ({total_bytes / (1024*1024):.1f} MB)")
    logger.info(f"Toplam Doküman Sayısı: {total_docs:,}")
    logger.info(f"Train Tokens         : {total_train_tokens:,} ({train_size_mb:.2f} MB -> {train_out_path})")
    logger.info(f"Val Tokens           : {total_val_tokens:,} ({val_size_mb:.2f} MB -> {val_out_path})")
    logger.info(f"Toplam Token Sayısı  : {total_tokens:,} (~{total_tokens/1e9:.2f}B token)")
    logger.info(f"Disk Kullanım Özeti  : Başlangıç: {init_free_gb:.2f} GB boş -> Bitiş: {final_free_gb:.2f} GB boş (Kullanılan: {init_free_gb - final_free_gb:.2f} GB)")
    logger.info(f"Toplam Geçen Süre    : {total_time:.1f} saniye ({total_time/60:.1f} dakika)")
    logger.info(f"Ortalama Verim       : {total_tokens / max(total_time, 0.1):,.0f} token/saniye")


if __name__ == "__main__":
    main()

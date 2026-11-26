"""Mergen-SNN — Korpusu 2B Token'a Tamamlama Scripti (bpe_train_v3c.bin).

Mevcut v3b verisini (1.804B train token) 2.000.000.000 token'a tamamlar:
1. Türkçe Wikipedia (wikimedia/wikipedia, 20231101.tr): ~60M token
2. Kod / Python (codeparrot/codeparrot-clean): ~136M token (kalan miktar)
Tüm veriler RAM'de stream edilip tokenize edilir ve uint16 binary olarak append edilir.
"""

import argparse
import gc
import os
from pathlib import Path
import shutil
import sys
import time

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from datasets import load_dataset
import numpy as np
from tokenizers import Tokenizer

from mergen.data.streaming_sources import clean_text
from mergen.utils.logger import get_logger

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = get_logger("mergen.extend_corpus_2b")

PROCESSED_DIR = repo_root / "data" / "processed"
V3B_BIN = PROCESSED_DIR / "bpe_train_v3b.bin"
V3C_BIN = PROCESSED_DIR / "bpe_train_v3c.bin"
TOK_PATH = PROCESSED_DIR / "tokenizer_bpe_v2.json"

TARGET_TOTAL_TOKENS = 2_000_000_000  # 2.0B token


def stream_wikipedia_tr():
    """Türkçe Wikipedia metin akışı."""
    logger.info("Wikimedia Wikipedia (20231101.tr) akışı başlatılıyor...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.tr", streaming=True, split="train")
    for item in ds:
        text = item.get("text") or item.get("content") or ""
        text = clean_text(text)
        if len(text.strip()) >= 50:
            yield text


def stream_codeparrot():
    """Codeparrot temiz kod metin akışı."""
    logger.info("Codeparrot (codeparrot-clean) akışı başlatılıyor...")
    ds = load_dataset("codeparrot/codeparrot-clean", streaming=True, split="train")
    for item in ds:
        text = item.get("content") or item.get("text") or item.get("code") or ""
        text = clean_text(text)
        if len(text.strip()) >= 50:
            yield text


def stream_fineweb_fallback():
    """FineWeb fallback metin akışı."""
    logger.info("FineWeb (sample-10BT) fallback akışı başlatılıyor...")
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", streaming=True, split="train")
    for item in ds:
        text = item.get("text") or ""
        text = clean_text(text)
        if len(text.strip()) >= 50:
            yield text


def main():
    parser = argparse.ArgumentParser(description="Korpusu 2B Token'a Tamamla")
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOTAL_TOKENS, help="Hedef toplam token sayısı (Varsayılan: 2,000,000,000)")
    parser.add_argument("--tr-target", type=int, default=60_000_000, help="Eklenecek TR Wikipedia token hedefi")
    parser.add_argument("--batch-size", type=int, default=128, help="Tokenize batch boyutu")
    parser.add_argument("--buffer-tokens", type=int, default=500_000, help="Diske flush buffer büyüklüğü")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not TOK_PATH.exists():
        logger.error(f"Tokenizer bulunamadı: {TOK_PATH}")
        sys.exit(1)

    logger.info(f"Tokenizer yükleniyor: {TOK_PATH}")
    tokenizer = Tokenizer.from_file(str(TOK_PATH))
    eot_id = tokenizer.token_to_id("<endoftext>") or tokenizer.token_to_id("<unk>") or 0

    # 1. v3c dosyasının varlığını ve mevcut token sayısını kontrol et
    if not V3C_BIN.exists():
        if not V3B_BIN.exists():
            logger.error(f"Kaynak v3b dosyası bulunamadı: {V3B_BIN}")
            sys.exit(1)
        logger.info(f"bpe_train_v3c.bin mevcut değil. v3b kopyalanıyor: {V3B_BIN} -> {V3C_BIN}")
        t_copy = time.time()
        shutil.copyfile(V3B_BIN, V3C_BIN)
        logger.info(f"Kopyalama tamamlandı ({time.time() - t_copy:.1f}s)")

    v3c_size = V3C_BIN.stat().st_size
    current_tokens = v3c_size // 2
    logger.info(f"Mevcut bpe_train_v3c.bin boyutu: {v3c_size:,} bayt ({v3c_size / (1024**3):.2f} GB)")
    logger.info(f"Mevcut Token Sayısı: {current_tokens:,} (~{current_tokens / 1e9:.3f}B token)")

    if current_tokens >= args.target_tokens:
        logger.info(f"✅ Dosya zaten hedefe ({args.target_tokens:,} token) ulaşmış durumda! İlave ekleme gerekmiyor.")
        return

    tokens_needed = args.target_tokens - current_tokens
    logger.info(f"2.0B Hedefi için Eksik Miktar: {tokens_needed:,} token (~{tokens_needed / 1e6:.1f}M token)")

    tr_tokens_goal = min(args.tr_target, tokens_needed)
    code_tokens_goal = max(tokens_needed - tr_tokens_goal, 0)
    logger.info(f"Planlanan Dağılım: TR Wikipedia: {tr_tokens_goal:,} token | Kod (Codeparrot): {code_tokens_goal:,} token")

    added_total = 0
    added_tr = 0
    added_code = 0
    t_start = time.time()

    with open(V3C_BIN, "ab") as f_out:
        buffer = []
        batch_texts = []

        def flush_batch():
            nonlocal buffer, batch_texts, added_total, added_tr, added_code
            if not batch_texts:
                return
            encs = tokenizer.encode_batch(batch_texts)
            batch_texts.clear()

            for enc, src in encs_with_src:
                ids = enc.ids
                if not ids:
                    continue
                doc_tokens = ids + [eot_id]
                rem = args.target_tokens - (current_tokens + added_total)
                if rem <= 0:
                    break
                if len(doc_tokens) > rem:
                    doc_tokens = doc_tokens[:rem]

                buffer.extend(doc_tokens)
                doc_len = len(doc_tokens)
                added_total += doc_len
                if src == "tr":
                    added_tr += doc_len
                else:
                    added_code += doc_len

            if len(buffer) >= args.buffer_tokens or (current_tokens + added_total >= args.target_tokens):
                arr = np.array(buffer, dtype=np.uint16)
                arr.tofile(f_out)
                f_out.flush()
                buffer.clear()

        # Aşama 1: Türkçe Wikipedia
        logger.info(f"\n--- Aşama 1: Türkçe Wikipedia Ekleme (Hedef: {tr_tokens_goal:,} token) ---")
        tr_stream = stream_wikipedia_tr()
        encs_with_src = []
        last_log_tok = 0

        for text in tr_stream:
            if added_tr >= tr_tokens_goal or (current_tokens + added_total >= args.target_tokens):
                break
            batch_texts.append(text)
            if len(batch_texts) >= args.batch_size:
                encs = tokenizer.encode_batch(batch_texts)
                encs_with_src = [(e, "tr") for e in encs]
                flush_batch()

            if added_total - last_log_tok >= 10_000_000 or (added_tr >= tr_tokens_goal):
                last_log_tok = added_total
                elapsed = time.time() - t_start
                tot_now = current_tokens + added_total
                pct = (tot_now / args.target_tokens) * 100
                speed = added_total / max(elapsed, 0.1)
                rem_tok = args.target_tokens - tot_now
                eta_s = rem_tok / max(speed, 1)
                logger.info(
                    f"[TR-Wiki] Eklenen: {added_total:,} tok | Toplam: {tot_now:,} / {args.target_tokens:,} "
                    f"(%{pct:.2f}) | Hız: {speed:,.0f} tok/s | ETA: {eta_s / 60:.1f} dk"
                )
                gc.collect()

        # Kalan batch'i boşalt
        if batch_texts:
            encs = tokenizer.encode_batch(batch_texts)
            encs_with_src = [(e, "tr") for e in encs]
            flush_batch()

        logger.info(f"Aşama 1 Tamamlandı. Eklenen TR token: {added_tr:,}")

        # Aşama 2: Kod (Codeparrot)
        code_needed = args.target_tokens - (current_tokens + added_total)
        logger.info(f"\n--- Aşama 2: Kod Ekleme (Kalan İhtiyaç: {code_needed:,} token) ---")
        if code_needed > 0:
            code_stream = stream_codeparrot()
            for text in code_stream:
                if current_tokens + added_total >= args.target_tokens:
                    break
                batch_texts.append(text)
                if len(batch_texts) >= args.batch_size:
                    encs = tokenizer.encode_batch(batch_texts)
                    encs_with_src = [(e, "code") for e in encs]
                    flush_batch()

                if added_total - last_log_tok >= 10_000_000:
                    last_log_tok = added_total
                    elapsed = time.time() - t_start
                    tot_now = current_tokens + added_total
                    pct = (tot_now / args.target_tokens) * 100
                    speed = added_total / max(elapsed, 0.1)
                    rem_tok = args.target_tokens - tot_now
                    eta_s = rem_tok / max(speed, 1)
                    logger.info(
                        f"[Codeparrot] Eklenen: {added_total:,} tok | Toplam: {tot_now:,} / {args.target_tokens:,} "
                        f"(%{pct:.2f}) | Hız: {speed:,.0f} tok/s | ETA: {eta_s / 60:.1f} dk"
                    )
                    gc.collect()

            if batch_texts:
                encs = tokenizer.encode_batch(batch_texts)
                encs_with_src = [(e, "code") for e in encs]
                flush_batch()

        # Buffer'da kalan varsa diske yaz
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            arr.tofile(f_out)
            f_out.flush()
            buffer.clear()

    total_time = time.time() - t_start
    final_size = V3C_BIN.stat().st_size
    final_tokens = final_size // 2

    logger.info("\n" + "=" * 60)
    logger.info("VERİ 2.0B TOKEN'A TAMAMLAMA İŞLEMİ BİTTİ!")
    logger.info("=" * 60)
    logger.info(f"Başlangıç Token Sayısı : {current_tokens:,}")
    logger.info(f"Eklenen TR Token       : {added_tr:,}")
    logger.info(f"Eklenen Kod Token      : {added_code:,}")
    logger.info(f"Toplam Eklenen Token   : {added_total:,}")
    logger.info(f"Son Toplam Token Sayısı: {final_tokens:,} (~{final_tokens / 1e9:.3f}B)")
    logger.info(f"Son Binary Dosya Boyutu: {final_size:,} bayt ({final_size / (1024**3):.2f} GB)")
    logger.info(f"Toplam Geçen Süre      : {total_time:.1f} saniye ({total_time / 60:.1f} dakika)")
    logger.info(f"Ortalama Hız           : {added_total / max(total_time, 0.1):,.0f} token/saniye")


if __name__ == "__main__":
    main()

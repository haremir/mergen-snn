"""Mergen-SNN — Disk Dostu (Streaming) Veri Toplama Kaynakları (v3 Dengeli Karışım).

Ham veriyi diske kaydetmeden doğrudan RAM'de temizler ve dengeli stream eder.
Yeni Oranlar:
- Kod (The Stack / codeparrot): %55 (~594M token)
- Türkçe Wikipedia (CulturaX / wikimedia): %15 (~162M token)
- FineWeb (genel web / CS): %15 (~162M token)
- S2ORC / ML-ArXiv (bilimsel CS): %10 (~108M token)
- Mevcut Korpus (yerel .txt): %5 (~54M token)
Toplam: 1.08 Milyar token (~3.66 GB metin).
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Generator, Tuple, Dict, Any, Optional

from mergen.utils.logger import get_logger

logger = get_logger("mergen.streaming_sources")

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

# 1.08B Token Hedef Kotaları
DEFAULT_TOKEN_QUOTAS = {
    "The Stack / Code": 594_000_000,   # %55
    "CulturaX / TR": 162_000_000,      # %15
    "FineWeb": 162_000_000,            # %15
    "S2ORC / ArXiv": 108_000_000,      # %10
    "Mevcut Local TXT": 54_000_000,    # %5
}

# Bayt Eşdeğerleri (~3.3 bayt/token)
DEFAULT_BYTE_QUOTAS = {
    "The Stack / Code": int(594_000_000 * 3.5),   # ~2079 MB
    "CulturaX / TR": int(162_000_000 * 3.0),      # ~486 MB
    "FineWeb": int(162_000_000 * 3.5),            # ~567 MB
    "S2ORC / ArXiv": int(108_000_000 * 3.6),      # ~388 MB
    "Mevcut Local TXT": int(54_000_000 * 3.3),    # ~178 MB
}


def clean_text(text: str) -> str:
    """Ligatürleri normalize et, non-breaking space dönüştür ve geçersiz kontrol karakterlerini ayıkla."""
    if not text:
        return ""
    for lig, repl in LIGATURE_MAP.items():
        if lig in text:
            text = text.replace(lig, repl)
    text = text.replace("\xa0", " ")
    allowed = {"\t", "\n", "\r"}
    cleaned = []
    for ch in text:
        cp = ord(ch)
        if ch in allowed or (32 <= cp <= 126) or (cp >= 160):
            cleaned.append(ch)
    return "".join(cleaned)


def stream_the_stack(target_bytes: int = DEFAULT_BYTE_QUOTAS["The Stack / Code"]) -> Generator[str, None, Tuple[int, int]]:
    """The Stack v2 / codeparrot üzerinden kod verisi akışı (%55)."""
    from datasets import load_dataset

    bytes_collected = 0
    doc_count = 0

    ds = None
    try:
        logger.info("The Stack v2 bağlanmaya çalışılıyor (bigcode/the-stack-v2)...")
        ds = load_dataset("bigcode/the-stack-v2", streaming=True, split="train")
        test_it = iter(ds)
        first = next(test_it)
        logger.info("bigcode/the-stack-v2 başarıyla bağlandı!")
    except Exception as e:
        logger.warning(f"SKIP: bigcode/the-stack-v2 gated/erişilemedi ({type(e).__name__}: {e}). Açık kod kaynağı 'codeparrot/codeparrot-clean' devreye alınıyor...")
        try:
            ds = load_dataset("codeparrot/codeparrot-clean", streaming=True, split="train")
            logger.info("codeparrot/codeparrot-clean akışı başlatıldı.")
        except Exception as e2:
            logger.error(f"HATA: Kod akışı başlatılamadı: {e2}")
            return bytes_collected, doc_count

    t0 = time.time()
    for item in ds:
        text = item.get("content") or item.get("text") or item.get("code") or ""
        text = clean_text(text)
        if len(text.strip()) < 50:
            continue

        encoded_len = len(text.encode("utf-8"))
        bytes_collected += encoded_len
        doc_count += 1
        yield text

        if doc_count % 1000 == 0:
            elapsed = time.time() - t0
            mb = bytes_collected / (1024 * 1024)
            logger.info(f"[The Stack/Code %55] {doc_count:,} döküman | {mb:.1f} MB / {target_bytes/(1024*1024):.0f} MB ({mb/(target_bytes/(1024*1024))*100:.1f}%) | {doc_count/max(elapsed, 0.1):.1f} doc/s")
            gc.collect()

        if bytes_collected >= target_bytes:
            break

    return bytes_collected, doc_count


def stream_culturax_tr(target_bytes: int = DEFAULT_BYTE_QUOTAS["CulturaX / TR"]) -> Generator[str, None, Tuple[int, int]]:
    """CulturaX Türkçe / Wikipedia TR metin akışı (%15)."""
    from datasets import load_dataset

    bytes_collected = 0
    doc_count = 0

    ds = None
    try:
        logger.info("CulturaX Türkçe bağlanmaya çalışılıyor (uonlp/CulturaX, subset tr)...")
        ds = load_dataset("uonlp/CulturaX", "tr", streaming=True, split="train")
        test_it = iter(ds)
        first = next(test_it)
        logger.info("uonlp/CulturaX başarıyla bağlandı!")
    except Exception as e:
        logger.warning(f"SKIP: uonlp/CulturaX gated/erişilemedi ({type(e).__name__}: {e}). Açık Türkçe kaynağı 'wikimedia/wikipedia (20231101.tr)' devreye alınıyor...")
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.tr", streaming=True, split="train")
            logger.info("wikimedia/wikipedia (20231101.tr) akışı başlatıldı.")
        except Exception as e2:
            logger.error(f"HATA: Türkçe akış başlatılamadı: {e2}")
            return bytes_collected, doc_count

    t0 = time.time()
    for item in ds:
        text = item.get("text") or item.get("content") or ""
        text = clean_text(text)
        if len(text.strip()) < 50:
            continue

        encoded_len = len(text.encode("utf-8"))
        bytes_collected += encoded_len
        doc_count += 1
        yield text

        if doc_count % 1000 == 0:
            elapsed = time.time() - t0
            mb = bytes_collected / (1024 * 1024)
            logger.info(f"[CulturaX/TR %15] {doc_count:,} döküman | {mb:.1f} MB / {target_bytes/(1024*1024):.0f} MB ({mb/(target_bytes/(1024*1024))*100:.1f}%) | {doc_count/max(elapsed, 0.1):.1f} doc/s")
            gc.collect()

        if bytes_collected >= target_bytes:
            break

    return bytes_collected, doc_count


def stream_s2orc(target_bytes: int = DEFAULT_BYTE_QUOTAS["S2ORC / ArXiv"]) -> Generator[str, None, Tuple[int, int]]:
    """S2ORC / ML-ArXiv CS makaleleri metin akışı (%10)."""
    from datasets import load_dataset

    bytes_collected = 0
    doc_count = 0

    ds = None
    try:
        logger.info("S2ORC bağlanmaya çalışılıyor (allenai/s2orc)...")
        ds = load_dataset("allenai/s2orc", streaming=True, split="train")
        test_it = iter(ds)
        first = next(test_it)
        logger.info("allenai/s2orc başarıyla bağlandı!")
    except Exception as e:
        logger.warning(f"SKIP: allenai/s2orc erişilemedi ({type(e).__name__}: {e}). Alternatif CS kaynağı 'CShorten/ML-ArXiv-Papers' devreye alınıyor...")
        try:
            ds = load_dataset("CShorten/ML-ArXiv-Papers", streaming=True, split="train")
            logger.info("CShorten/ML-ArXiv-Papers akışı başlatıldı.")
        except Exception as e2:
            logger.error(f"HATA: CS makale akışı başlatılamadı: {e2}")
            return bytes_collected, doc_count

    t0 = time.time()
    for item in ds:
        title = item.get("title") or ""
        abstract = item.get("abstract") or item.get("text") or item.get("body") or ""
        text = f"{title}\n\n{abstract}" if title and abstract != title else abstract
        text = clean_text(text)
        if len(text.strip()) < 50:
            continue

        encoded_len = len(text.encode("utf-8"))
        bytes_collected += encoded_len
        doc_count += 1
        yield text

        if doc_count % 1000 == 0:
            elapsed = time.time() - t0
            mb = bytes_collected / (1024 * 1024)
            logger.info(f"[S2ORC/ArXiv %10] {doc_count:,} döküman | {mb:.1f} MB / {target_bytes/(1024*1024):.0f} MB ({mb/(target_bytes/(1024*1024))*100:.1f}%) | {doc_count/max(elapsed, 0.1):.1f} doc/s")
            gc.collect()

        if bytes_collected >= target_bytes:
            break

    return bytes_collected, doc_count


def stream_fineweb(target_bytes: int = DEFAULT_BYTE_QUOTAS["FineWeb"]) -> Generator[str, None, Tuple[int, int]]:
    """FineWeb genel İngilizce/CS metin akışı (%15)."""
    from datasets import load_dataset

    bytes_collected = 0
    doc_count = 0

    try:
        logger.info("FineWeb bağlanıyor (HuggingFaceFW/fineweb, subset sample-10BT)...")
        ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", streaming=True, split="train")
        logger.info("FineWeb başarıyla bağlandı!")
    except Exception as e:
        logger.error(f"HATA: FineWeb bağlanamadı: {e}")
        return bytes_collected, doc_count

    t0 = time.time()
    for item in ds:
        text = item.get("text") or ""
        text = clean_text(text)
        if len(text.strip()) < 50:
            continue

        encoded_len = len(text.encode("utf-8"))
        bytes_collected += encoded_len
        doc_count += 1
        yield text

        if doc_count % 1000 == 0:
            elapsed = time.time() - t0
            mb = bytes_collected / (1024 * 1024)
            logger.info(f"[FineWeb %15] {doc_count:,} döküman | {mb:.1f} MB / {target_bytes/(1024*1024):.0f} MB ({mb/(target_bytes/(1024*1024))*100:.1f}%) | {doc_count/max(elapsed, 0.1):.1f} doc/s")
            gc.collect()

        if bytes_collected >= target_bytes:
            break

    return bytes_collected, doc_count


def stream_local_txt(raw_dir: Path = Path("data/raw/cs_corpus"), target_bytes: int = DEFAULT_BYTE_QUOTAS["Mevcut Local TXT"]) -> Generator[str, None, Tuple[int, int]]:
    """data/raw/cs_corpus/ altındaki mevcut .txt dosyalarından metin akışı (%5)."""
    bytes_collected = 0
    doc_count = 0

    if not raw_dir.exists():
        logger.warning(f"Yerel raw dizini bulunamadı: {raw_dir}")
        return bytes_collected, doc_count

    txt_files = sorted(raw_dir.rglob("*.txt"))
    logger.info(f"Yerel korpus taranıyor: {len(txt_files)} dosya bulundu.")
    t0 = time.time()

    for fpath in txt_files:
        if fpath.name.startswith(("_", ".")) or fpath.name == "LICENSE_NOTE.txt":
            continue
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
            content = clean_text(content)
            if len(content.strip()) < 50:
                continue

            encoded_len = len(content.encode("utf-8"))
            bytes_collected += encoded_len
            doc_count += 1
            yield content

            if doc_count % 1000 == 0:
                elapsed = time.time() - t0
                mb = bytes_collected / (1024 * 1024)
                logger.info(f"[Local TXT %5] {doc_count:,} döküman | {mb:.1f} MB / {target_bytes/(1024*1024):.0f} MB | {doc_count/max(elapsed, 0.1):.1f} doc/s")
                gc.collect()

            if bytes_collected >= target_bytes:
                break
        except Exception:
            continue

    return bytes_collected, doc_count


def stream_all_sources(quotas: Dict[str, int] = None, interleave: bool = True) -> Generator[Tuple[str, str], None, None]:
    """Tüm 5 kaynaktan yeni dengeli oranlarla (%55 Kod, %15 TR, %15 FineWeb, %10 ArXiv, %5 Local) (kaynak_adı, metin) ikilileri üretir.
    
    interleave=True ise kaynaklar 11:3:3:2:1 oranında dönüşümlü (round-robin) çekilir.
    Böylece veri kümesinin her diliminde kod ve diğer tüm alanlar homojen olarak dağılır.
    """
    if quotas is None:
        quotas = DEFAULT_BYTE_QUOTAS

    if not interleave:
        # Kod öncelikli sıralı akış
        for text in stream_the_stack(target_bytes=quotas.get("The Stack / Code", DEFAULT_BYTE_QUOTAS["The Stack / Code"])):
            yield "The Stack / Code", text

        for text in stream_culturax_tr(target_bytes=quotas.get("CulturaX / TR", DEFAULT_BYTE_QUOTAS["CulturaX / TR"])):
            yield "CulturaX / TR", text

        for text in stream_fineweb(target_bytes=quotas.get("FineWeb", DEFAULT_BYTE_QUOTAS["FineWeb"])):
            yield "FineWeb", text

        for text in stream_s2orc(target_bytes=quotas.get("S2ORC / ArXiv", DEFAULT_BYTE_QUOTAS["S2ORC / ArXiv"])):
            yield "S2ORC / ArXiv", text

        for text in stream_local_txt(target_bytes=quotas.get("Mevcut Local TXT", DEFAULT_BYTE_QUOTAS["Mevcut Local TXT"])):
            yield "Mevcut Local TXT", text
        return

    # Homojen Dönüşümlü (Round-Robin Interleaved) Akış: 11 Kod, 3 TR, 3 FineWeb, 2 ArXiv, 1 Local
    generators = {
        "The Stack / Code": iter(stream_the_stack(target_bytes=quotas.get("The Stack / Code", DEFAULT_BYTE_QUOTAS["The Stack / Code"]))),
        "CulturaX / TR": iter(stream_culturax_tr(target_bytes=quotas.get("CulturaX / TR", DEFAULT_BYTE_QUOTAS["CulturaX / TR"]))),
        "FineWeb": iter(stream_fineweb(target_bytes=quotas.get("FineWeb", DEFAULT_BYTE_QUOTAS["FineWeb"]))),
        "S2ORC / ArXiv": iter(stream_s2orc(target_bytes=quotas.get("S2ORC / ArXiv", DEFAULT_BYTE_QUOTAS["S2ORC / ArXiv"]))),
        "Mevcut Local TXT": iter(stream_local_txt(target_bytes=quotas.get("Mevcut Local TXT", DEFAULT_BYTE_QUOTAS["Mevcut Local TXT"]))),
    }

    # Her döngüde çekilecek parça sayıları (Toplam 20 parça: 11/20=%55, 3/20=%15, 3/20=%15, 2/20=%10, 1/20=%5)
    ratios = {
        "The Stack / Code": 11,
        "CulturaX / TR": 3,
        "FineWeb": 3,
        "S2ORC / ArXiv": 2,
        "Mevcut Local TXT": 1,
    }

    active_sources = set(generators.keys())

    while active_sources:
        for src in list(active_sources):
            gen = generators[src]
            take_count = ratios[src]
            for _ in range(take_count):
                try:
                    text = next(gen)
                    yield src, text
                except StopIteration:
                    logger.info(f"Kaynak tamamlandı: {src}")
                    active_sources.discard(src)
                    break

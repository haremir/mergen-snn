"""Mergen-SNN — Disk Dostu (Streaming) Veri Toplama Kaynakları.

Ham veriyi diske kaydetmeden doğrudan RAM'de temizler ve stream eder.
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Generator, Tuple, Dict, Any

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


def stream_the_stack(target_bytes: int = 1500 * 1024 * 1024) -> Generator[str, None, Tuple[int, int]]:
    """The Stack v2 (veya açık kod alternatifi codeparrot) üzerinden kod verisi akışı."""
    from datasets import load_dataset

    bytes_collected = 0
    doc_count = 0
    
    # 1. İlk olarak kullanıcının belirttiği bigcode/the-stack-v2'yi dene
    ds = None
    try:
        logger.info("The Stack v2 bağlanmaya çalışılıyor (bigcode/the-stack-v2)...")
        ds = load_dataset("bigcode/the-stack-v2", streaming=True, split="train")
        # Test iterator
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
            logger.info(f"[The Stack/Code] {doc_count:,} döküman | {mb:.1f} MB / {target_bytes/(1024*1024):.0f} MB ({mb/(target_bytes/(1024*1024))*100:.1f}%) | {doc_count/max(elapsed, 0.1):.1f} doc/s")
            gc.collect()

        if bytes_collected >= target_bytes:
            break

    return bytes_collected, doc_count


def stream_culturax_tr(target_bytes: int = 1000 * 1024 * 1024) -> Generator[str, None, Tuple[int, int]]:
    """CulturaX Türkçe (veya açık Türkçe Wikipedia) metin akışı."""
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
            logger.info(f"[CulturaX/TR] {doc_count:,} döküman | {mb:.1f} MB / {target_bytes/(1024*1024):.0f} MB ({mb/(target_bytes/(1024*1024))*100:.1f}%) | {doc_count/max(elapsed, 0.1):.1f} doc/s")
            gc.collect()

        if bytes_collected >= target_bytes:
            break

    return bytes_collected, doc_count


def stream_s2orc(target_bytes: int = 1000 * 1024 * 1024) -> Generator[str, None, Tuple[int, int]]:
    """S2ORC CS makaleleri (veya ML-ArXiv-Papers) metin akışı."""
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
        logger.warning(f"SKIP: allenai/s2orc doğrudan streaming desteklemiyor/erişilemedi ({type(e).__name__}: {e}). Alternatif CS kaynağı 'CShorten/ML-ArXiv-Papers' devreye alınıyor...")
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
            logger.info(f"[S2ORC/ArXiv] {doc_count:,} döküman | {mb:.1f} MB / {target_bytes/(1024*1024):.0f} MB ({mb/(target_bytes/(1024*1024))*100:.1f}%) | {doc_count/max(elapsed, 0.1):.1f} doc/s")
            gc.collect()

        if bytes_collected >= target_bytes:
            break

    return bytes_collected, doc_count


def stream_fineweb(target_bytes: int = 1000 * 1024 * 1024) -> Generator[str, None, Tuple[int, int]]:
    """FineWeb genel İngilizce metin akışı."""
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
            logger.info(f"[FineWeb] {doc_count:,} döküman | {mb:.1f} MB / {target_bytes/(1024*1024):.0f} MB ({mb/(target_bytes/(1024*1024))*100:.1f}%) | {doc_count/max(elapsed, 0.1):.1f} doc/s")
            gc.collect()

        if bytes_collected >= target_bytes:
            break

    return bytes_collected, doc_count


def stream_local_txt(raw_dir: Path = Path("data/raw/cs_corpus"), target_bytes: int = 500 * 1024 * 1024) -> Generator[str, None, Tuple[int, int]]:
    """data/raw/cs_corpus/ altındaki mevcut .txt dosyalarından metin akışı."""
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
                logger.info(f"[Local TXT] {doc_count:,} döküman | {mb:.1f} MB / {target_bytes/(1024*1024):.0f} MB | {doc_count/max(elapsed, 0.1):.1f} doc/s")
                gc.collect()

            if bytes_collected >= target_bytes:
                break
        except Exception as e:
            continue

    return bytes_collected, doc_count


def stream_all_sources(quotas: Dict[str, int] = None) -> Generator[Tuple[str, str], None, None]:
    """Tüm 5 kaynaktan sırayla (kaynak_adı, metin) ikilileri üretir."""
    if quotas is None:
        quotas = {
            "The Stack / Code": 1500 * 1024 * 1024,
            "CulturaX / TR": 1000 * 1024 * 1024,
            "S2ORC / ArXiv": 1000 * 1024 * 1024,
            "FineWeb": 1000 * 1024 * 1024,
            "Mevcut Local TXT": 500 * 1024 * 1024,
        }

    # 1. Local TXT
    for text in stream_local_txt(target_bytes=quotas.get("Mevcut Local TXT", 500 * 1024 * 1024)):
        yield "Mevcut Local TXT", text

    # 2. CulturaX / Turkish
    for text in stream_culturax_tr(target_bytes=quotas.get("CulturaX / TR", 1000 * 1024 * 1024)):
        yield "CulturaX / TR", text

    # 3. The Stack / Code
    for text in stream_the_stack(target_bytes=quotas.get("The Stack / Code", 1500 * 1024 * 1024)):
        yield "The Stack / Code", text

    # 4. S2ORC / CS Papers
    for text in stream_s2orc(target_bytes=quotas.get("S2ORC / ArXiv", 1000 * 1024 * 1024)):
        yield "S2ORC / ArXiv", text

    # 5. FineWeb
    for text in stream_fineweb(target_bytes=quotas.get("FineWeb", 1000 * 1024 * 1024)):
        yield "FineWeb", text

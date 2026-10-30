"""Mergen-SNN — Massive Corpus Streaming Pipeline (Disk Dostu).

Ham veriyi diske yazmadan doğrudan RAM'de stream eder.
"""

import gc
import sys
import time
from mergen.data.streaming_sources import stream_all_sources
from mergen.utils.logger import get_logger

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = get_logger("mergen.fetch_massive_corpus")


def main():
    logger.info("==========================================")
    logger.info("Mergen-SNN — Disk Dostu Streaming Veri Akışı Testi")
    logger.info("==========================================")

    # Test modunda her kaynaktan küçük bir kota test edilir
    test_quotas = {
        "Mevcut Local TXT": 5 * 1024 * 1024,
        "CulturaX / TR": 10 * 1024 * 1024,
        "The Stack / Code": 15 * 1024 * 1024,
        "S2ORC / ArXiv": 10 * 1024 * 1024,
        "FineWeb": 10 * 1024 * 1024,
    }

    stats = {k: {"bytes": 0, "docs": 0} for k in test_quotas}
    t_start = time.time()

    for source_name, text in stream_all_sources(quotas=test_quotas):
        b_len = len(text.encode("utf-8"))
        stats[source_name]["bytes"] += b_len
        stats[source_name]["docs"] += 1

    total_time = time.time() - t_start
    total_bytes = sum(s["bytes"] for s in stats.values())
    total_docs = sum(s["docs"] for s in stats.values())

    logger.info("\n--- Streaming Doğrulama Özeti ---")
    for src, d in stats.items():
        mb = d["bytes"] / (1024 * 1024)
        logger.info(f"  {src:<22}: {d['docs']:>8,} döküman | {mb:>8.2f} MB")
    logger.info(f"Toplam Akış: {total_docs:,} döküman | {total_bytes/(1024*1024):.2f} MB | {total_time:.1f} saniye")


if __name__ == "__main__":
    main()

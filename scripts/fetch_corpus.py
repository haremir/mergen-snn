import os
from pathlib import Path
from urllib.parse import urlparse
from mergen.utils.logger import get_logger

try:
    import requests
except ImportError:
    requests = None
    import urllib.request

logger = get_logger("mergen.fetch_corpus")

BASE_DIR = Path("data/raw/cs_corpus")

# Kategori -> İndirilecek URL listesi
SOURCES = {
    "ostep": [
        "https://pages.cs.wisc.edu/~remzi/OSTEP/",
    ],
    "cython_docs": [
        "https://cython.readthedocs.io/en/latest/",
    ],
    "gc_papers": [
        "https://raw.githubusercontent.com/papers-we-love/papers-we-love/master/garbage-collection/README.md",
    ],
    "python_docs": [],
    "c_docs": [],
    "cpp_docs": [],
    "architecture": [],
    "systems": [],
    "unix_os": [],
}


def download_url(url: str, dest_dir: Path) -> bool:
    """Belirtilen URL'yi hedef dizine indirir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path) or "index.html"
    dest_file = dest_dir / filename

    logger.info(f"İndiriliyor: {url} -> {dest_file}")

    try:
        if requests is not None:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            content = resp.content
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "Mergen-SNN-Corpus-Fetcher/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                content = r.read()

        with open(dest_file, "wb") as f:
            f.write(content)

        logger.info(f"Başarıyla kaydedildi: {dest_file} ({len(content)} byte)")
        return True

    except Exception as e:
        logger.error(f"İndirme başarısız oldu ({url}): {e}")
        return False


def fetch_all():
    """Tüm kategoriler için tanımlı veri kaynaklarını indirir."""
    logger.info("==========================================")
    logger.info("Mergen-SNN — CS Korpusu İndirme İskeleti")
    logger.info("==========================================")

    for category, urls in SOURCES.items():
        cat_dir = BASE_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        if not urls:
            logger.info(f"Kategori '{category}' için henüz URL tanımlanmadı (boş).")
            continue

        for url in urls:
            download_url(url, cat_dir)


if __name__ == "__main__":
    fetch_all()

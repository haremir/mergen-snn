import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import pypdf
import requests

from mergen.utils.logger import get_logger

logger = get_logger("mergen.fetch_corpus")

BASE_DIR = Path("data/raw/cs_corpus")

# ==============================================================================
# 1. Veri Kaynakları Tanımları
# ==============================================================================

DOWNLOADABLE_SOURCES: Dict[str, Dict] = {
    "ostep": {
        "license": "Free to read online, © Remzi & Andrea Arpaci-Dusseau. Personal educational use only.",
        "files": {
            "cpu-intro.pdf": "https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-intro.pdf",
            "cpu-mechanisms.pdf": "https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-mechanisms.pdf",
            "cpu-sched.pdf": "https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-sched.pdf",
            "vm-intro.pdf": "https://pages.cs.wisc.edu/~remzi/OSTEP/vm-intro.pdf",
            "vm-paging.pdf": "https://pages.cs.wisc.edu/~remzi/OSTEP/vm-paging.pdf",
            "threads-intro.pdf": "https://pages.cs.wisc.edu/~remzi/OSTEP/threads-intro.pdf",
            "file-intro.pdf": "https://pages.cs.wisc.edu/~remzi/OSTEP/file-intro.pdf",
        },
    },
    "python_docs": {
        "license": "Python Software Foundation License (PSF-2.0).",
        "files": {
            "tutorial_index.html": "https://docs.python.org/3/tutorial/index.html",
            "tutorial_introduction.html": "https://docs.python.org/3/tutorial/introduction.html",
            "tutorial_controlflow.html": "https://docs.python.org/3/tutorial/controlflow.html",
            "tutorial_datastructures.html": "https://docs.python.org/3/tutorial/datastructures.html",
            "tutorial_modules.html": "https://docs.python.org/3/tutorial/modules.html",
            "tutorial_classes.html": "https://docs.python.org/3/tutorial/classes.html",
        },
    },
    "cython_docs": {
        "license": "Apache-2.0 (Cython project).",
        "files": {
            "cython_tutorial.html": "https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html",
            "cython_cdef.html": "https://cython.readthedocs.io/en/latest/src/tutorial/cdef.html",
            "cython_strings.html": "https://cython.readthedocs.io/en/latest/src/tutorial/strings.html",
            "cython_pure.html": "https://cython.readthedocs.io/en/latest/src/tutorial/pure.html",
        },
    },
}

GIT_CLONE_SOURCES: Dict[str, Dict] = {
    "gc_papers": {
        "repo": "https://github.com/papers-we-love/papers-we-love.git",
        "subpath": "garbage-collection",
        "license": "Papers hosted for educational purposes under fair use; original authors retain copyright.",
    }
}

LOCAL_REQUIRED_SOURCES: Dict[str, Dict] = {
    "architecture_local": {
        "files": [
            "patterson_hennessy_cod.pdf",
            "patterson_hennessy_caa.pdf",
        ],
        "instruction": "Bu telifli kitaplar otomatik indirilemez. PDF'lerini data/raw/cs_corpus/architecture_local/ altına elle koy, sonra bu script'i tekrar çalıştır.",
        "license": "Copyrighted textbooks (Patterson & Hennessy). Local copy provided by user for personal research.",
    }
}


# ==============================================================================
# 2. Yardımcı Fonksiyonlar (Metin Çıkarma ve Dosya İşlemleri)
# ==============================================================================

def remove_readonly(func, path, _):
    """Windows'ta git reposu silinirken salt-okunur izinlerini kaldırır."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def write_license_note(dest_dir: Path, note: str):
    """Kategori dizinine LICENSE_NOTE.txt dosyasını yazar."""
    license_file = dest_dir / "LICENSE_NOTE.txt"
    with open(license_file, "w", encoding="utf-8") as f:
        f.write(note.strip() + "\n")


def extract_text_from_html(html_path: Path) -> bool:
    """HTML dosyasından script/style/nav etiketlerini temizleyerek .txt üretir."""
    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "lxml")
        for tag in soup(["script", "style", "nav", "aside", "header", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        clean_text = "\n".join(chunk for chunk in lines if chunk)

        txt_path = html_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(clean_text)

        return True
    except Exception as e:
        logger.error(f"[SKIP] HTML extraction hatası ({html_path.name}): {e}")
        return False


def extract_text_from_pdf(pdf_path: Path) -> bool:
    """PDF dosyasından pypdf ile sayfa sayfa metin çıkararak .txt üretir."""
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        pages_text = []
        for page_idx, page in enumerate(reader.pages):
            try:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
            except Exception as pe:
                logger.warning(f"{pdf_path.name} sayfa {page_idx} çıkarılamadı: {pe}")

        if not pages_text:
            logger.warning(f"[SKIP] {pdf_path.name} içinde metin bulunamadı.")
            return False

        txt_path = pdf_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(pages_text))

        return True
    except Exception as e:
        logger.error(f"[SKIP] PDF extraction hatası ({pdf_path.name}): {e}")
        return False


# ==============================================================================
# 3. İndirme ve İşleme Yöneticisi
# ==============================================================================

def format_size(bytes_num: int) -> str:
    """Byte miktarını okunabilir boyuta dönüştürür."""
    if bytes_num < 1024:
        return f"{bytes_num} B"
    elif bytes_num < 1024 * 1024:
        return f"{bytes_num / 1024:.1f} KB"
    else:
        return f"{bytes_num / (1024 * 1024):.2f} MB"


def main():
    logger.info("==========================================")
    logger.info("Mergen-SNN — CS Korpus İndirme & Metin Çıkarma")
    logger.info("==========================================")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })

    stats: Dict[str, Dict] = {}
    skip_list: List[Dict] = []
    pdf_stats = {"total": 0, "success": 0}

    # --------------------------------------------------------------------------
    # A) İndirilebilir Kaynaklar (Downloadable Sources)
    # --------------------------------------------------------------------------
    for category, cat_data in DOWNLOADABLE_SOURCES.items():
        cat_dir = BASE_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        write_license_note(cat_dir, cat_data["license"])

        cat_stats = {"downloaded": 0, "bytes": 0, "txt_extracted": 0, "errors": 0}

        for filename, url in cat_data["files"].items():
            dest_file = cat_dir / filename
            try:
                resp = session.get(url, timeout=30)
                if resp.status_code != 200:
                    err_msg = f"HTTP {resp.status_code}"
                    print(f"[SKIP] {category}/{url} ({err_msg})")
                    logger.warning(f"[SKIP] {category}/{url} ({err_msg})")
                    cat_stats["errors"] += 1
                    skip_list.append({"category": category, "url": url, "reason": err_msg})
                    continue

                content = resp.content
                with open(dest_file, "wb") as f:
                    f.write(content)

                cat_stats["downloaded"] += 1
                cat_stats["bytes"] += len(content)
                size_str = format_size(len(content))
                print(f"[OK] {category}/{filename} ({size_str})")
                logger.info(f"[OK] {category}/{filename} ({size_str})")

                # Metin çıkarma
                ext = dest_file.suffix.lower()
                if ext == ".html":
                    if extract_text_from_html(dest_file):
                        cat_stats["txt_extracted"] += 1
                elif ext == ".pdf":
                    pdf_stats["total"] += 1
                    if extract_text_from_pdf(dest_file):
                        cat_stats["txt_extracted"] += 1
                        pdf_stats["success"] += 1

            except Exception as e:
                err_msg = str(e)
                print(f"[SKIP] {category}/{url} ({err_msg})")
                logger.warning(f"[SKIP] {category}/{url} ({err_msg})")
                cat_stats["errors"] += 1
                skip_list.append({"category": category, "url": url, "reason": err_msg})

        stats[category] = cat_stats

    # --------------------------------------------------------------------------
    # B) Git Klonlama Kaynakları (Git Clone Sources)
    # --------------------------------------------------------------------------
    for category, cat_data in GIT_CLONE_SOURCES.items():
        cat_dir = BASE_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        write_license_note(cat_dir, cat_data["license"])

        cat_stats = {"downloaded": 0, "bytes": 0, "txt_extracted": 0, "errors": 0}
        repo_url = cat_data["repo"]
        subpath = cat_data["subpath"]

        logger.info(f"Git repo klonlanıyor (sparse): {repo_url} [{subpath}]...")
        temp_dir = tempfile.mkdtemp(prefix="mergen_git_")

        try:
            # Hızlı sparse clone
            cmd_clone = ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", repo_url, temp_dir]
            subprocess.run(cmd_clone, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            cmd_sparse = ["git", "-C", temp_dir, "sparse-checkout", "set", subpath]
            subprocess.run(cmd_sparse, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            src_folder = Path(temp_dir) / subpath
            if not src_folder.exists():
                # İsim alternatiflerini dene
                for alt in ["garbage_collection", "garbage-collection", "Garbage-Collection"]:
                    candidate = Path(temp_dir) / alt
                    if candidate.exists():
                        src_folder = candidate
                        break

            if src_folder.exists():
                for item in src_folder.iterdir():
                    if item.is_file():
                        dest_file = cat_dir / item.name
                        shutil.copy2(item, dest_file)
                        cat_stats["downloaded"] += 1
                        size = dest_file.stat().st_size
                        cat_stats["bytes"] += size
                        print(f"[OK] {category}/{item.name} ({format_size(size)})")

                        ext = dest_file.suffix.lower()
                        if ext == ".pdf":
                            pdf_stats["total"] += 1
                            if extract_text_from_pdf(dest_file):
                                cat_stats["txt_extracted"] += 1
                                pdf_stats["success"] += 1
                        elif ext in [".md", ".txt"]:
                            # Zaten metin dosyası
                            cat_stats["txt_extracted"] += 1
            else:
                msg = f"Repo içinde {subpath} klasörü bulunamadı."
                print(f"[SKIP] {category}/{repo_url} ({msg})")
                cat_stats["errors"] += 1
                skip_list.append({"category": category, "url": repo_url, "reason": msg})

        except Exception as e:
            msg = f"Git işlemi başarısız: {e}"
            print(f"[SKIP] {category}/{repo_url} ({msg})")
            cat_stats["errors"] += 1
            skip_list.append({"category": category, "url": repo_url, "reason": msg})

        finally:
            shutil.rmtree(temp_dir, onerror=remove_readonly)

        stats[category] = cat_stats

    # --------------------------------------------------------------------------
    # C) Yerel Zorunlu Kaynaklar (Local Required Sources)
    # --------------------------------------------------------------------------
    for category, cat_data in LOCAL_REQUIRED_SOURCES.items():
        cat_dir = BASE_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        write_license_note(cat_dir, cat_data["license"])

        cat_stats = {"downloaded": 0, "bytes": 0, "txt_extracted": 0, "errors": 0, "missing": 0}

        for fname in cat_data["files"]:
            fpath = cat_dir / fname
            if fpath.exists():
                size = fpath.stat().st_size
                cat_stats["downloaded"] += 1
                cat_stats["bytes"] += size
                print(f"[OK] {category}/{fname} ({format_size(size)})")
                pdf_stats["total"] += 1
                if extract_text_from_pdf(fpath):
                    cat_stats["txt_extracted"] += 1
                    pdf_stats["success"] += 1
            else:
                cat_stats["missing"] += 1
                print(f"[EKSİK] {category}/{fname}")
                print(f"       Talimat: {cat_data['instruction']}")

        stats[category] = cat_stats

    # --------------------------------------------------------------------------
    # D) Genel Özet Raporu
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("KATEGORİ BAZINDA KORPUS ÖZETİ")
    print("=" * 60)
    for cat, data in stats.items():
        err_str = f", Hata/404: {data['errors']}" if "errors" in data else ""
        missing_str = f", Eksik: {data['missing']}" if "missing" in data else ""
        print(
            f"- {cat:<20}: {data['downloaded']} dosya ({format_size(data['bytes'])}), "
            f"Çıkarılan .txt: {data['txt_extracted']}{err_str}{missing_str}"
        )

    print("-" * 60)
    print(f"Toplam Atlanan/Hata (SKIP) Sayısı: {len(skip_list)}")
    if pdf_stats["total"] > 0:
        ratio = (pdf_stats["success"] / pdf_stats["total"]) * 100
        print(f"PDF Metin Çıkarma Başarısı       : {pdf_stats['success']}/{pdf_stats['total']} (%{ratio:.1f})")
    else:
        print("PDF Metin Çıkarma Başarısı       : PDF işlenmedi")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

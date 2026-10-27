import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import pypdf
import requests

from mergen.utils.logger import get_logger

logger = get_logger("mergen.fetch_corpus")

BASE_DIR = Path("data/raw/cs_corpus")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# ==============================================================================
# 1. Kaynak ve Kategori Tanımları
# ==============================================================================

LICENSE_NOTES = {
    "architecture": "Computer Architecture reference textbooks and open ISA specifications (Patterson & Hennessy, RISC-V International).",
    "algorithms": "Open educational resources and open-access algorithmic textbooks (Jeff Erickson, AlgoXY, OpenDataStructures).",
    "theory": "Open textbooks in theoretical computer science and discrete mathematics (MIT OCW, HWS, StFX).",
    "systems": "Open systems textbooks and reference materials (Dive into Systems, Bottom Up CS, Gustavus OS).",
    "compilers": "Open compiler textbooks (Crafting Interpreters by Bob Nystrom, Basics of Compiler Design by Torben Mogensen).",
    "databases": "PostgreSQL Reference Documentation and Readings in Database Systems (The Red Book 5th ed.).",
    "networking": "Open Computer Networking principles and foundational IETF RFC standards (IETF / computer-networking.info).",
    "ai_ml": "Open deep learning and reinforcement learning books (Dive into Deep Learning - D2L, Sutton & Barto RL).",
    "papers_we_love": "Academic CS papers curated by the Papers We Love community, hosted for educational research.",
    "lecture_notes": "Open courseware and university lecture slides (CMU 15-213, MIT OCW).",
    "ostep": "Free to read online, © Remzi & Andrea Arpaci-Dusseau. Personal educational use only.",
    "python_docs": "Python Software Foundation License (PSF-2.0).",
    "c_docs": "ISO C Standards Draft (WG14) and GNU C Reference Manual.",
    "cpp_docs": "ISO C++ Standards Draft (WG21 - Standard for Programming Language C++).",
    "cython_docs": "Apache-2.0 (Cython project).",
    "architecture_local": "Copyrighted textbooks (Patterson & Hennessy). Local copy provided by user for personal research.",
}

# Doğrudan İndirilecek Büyük Dosyalar (PDF / TXT)
DIRECT_PDF_SOURCES = {
    "architecture": [
        # 1. Patterson & Hennessy 5. Baskı (Google Drive)
        ("patterson_hennessy_cod_5th.pdf", "https://docs.google.com/uc?export=download&id=1OwVqBQkyjW4xvUvrVYyx3zMe5XkRHRlX"),
        # 2. Patterson & Hennessy 6. Baskı (Archive.org)
        ("patterson_hennessy_caa_6th.pdf", "https://archive.org/download/computerarchitectureaquantitativeapproach6thedition/Computer%20Architecture%3A%20A%20Quantitative%20Approach%206th%20Edition.pdf"),
        # 3. Patterson & Hennessy 4. Baskı (GitHub raw)
        ("patterson_hennessy_caa_4th.pdf", "https://github.com/lovisticsdev/engineering_books/raw/main/Computer%20Architecture/Hennessy%2C%20Patterson%20-%20Computer%20Architecture%20A%20Quantitative%20Approach%20%5B4th%20Edition%5D.pdf"),
        # 4. Patterson & Hennessy RISC-V Edition (GitHub raw)
        ("patterson_hennessy_cod_riscv.pdf", "https://github.com/kk-1024b/CS-textbooks/raw/main/Computer-Organization-and-Design-The-Hardware-Software-Interface-RISC-V-Edition.pdf"),
        # Açık Erişim RISC-V Spesifikasyonları ve Mimarisi
        ("riscv_unprivileged_spec.pdf", "https://github.com/riscv/riscv-isa-manual/releases/download/Ratified-IMAFDQC/riscv-spec-20191213.pdf"),
        ("riscv_privileged_spec.pdf", "https://github.com/riscv/riscv-isa-manual/releases/download/Priv-v1.12/riscv-privileged-20211203.pdf"),
        ("xv6_riscv_architecture_book.pdf", "https://pdos.csail.mit.edu/6.828/2023/xv6/book-riscv-rev3.pdf"),
    ],
    "cpp_docs": [
        ("cpp23_standard_draft_n4950.pdf", "https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/n4950.pdf"),
    ],
    "databases": [
        ("postgresql_16_documentation.pdf", "https://www.postgresql.org/files/documentation/pdf/16/postgresql-16-A4.pdf"),
    ],
    "c_docs": [
        ("c11_standard_n1570.pdf", "https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf"),
    ],
    "algorithms": [
        ("algorithms_jeffe.pdf", "https://jeffe.cs.illinois.edu/teaching/algorithms/book/Algorithms-JeffE.pdf"),
        ("elementary_algorithms.pdf", "https://github.com/liuxinyu95/AlgoXY/releases/download/v0.618033/elementary-algorithms.pdf"),
        ("ods_python.pdf", "https://opendatastructures.org/ods-python.pdf"),
    ],
    "theory": [
        ("theory_of_computing.pdf", "https://people.stfx.ca/tasmith/theory/TheoryOfComputing.pdf"),
        ("foundations_of_computation.pdf", "https://math.hws.edu/FoundationsOfComputation/FoundationsOfComputation.pdf"),
        ("mit_mcs.pdf", "https://courses.csail.mit.edu/6.042/spring18/mcs.pdf"),
    ],
    "systems": [
        ("os_book_max.pdf", "https://gustavus.edu/mcs/max/os-book/os-book.pdf"),
        ("dive_into_systems.pdf", "https://diveintosystems.org/book/pdf/DiveIntoSystems.pdf"),
        ("progit_2nd_edition.pdf", "https://github.com/progit/progit2/releases/download/2.1.423/progit.pdf"),
        ("gdb_debugging_manual.pdf", "https://sourceware.org/gdb/current/onlinedocs/gdb.pdf"),
    ],
    "compilers": [
        ("gcc_manual.pdf", "https://gcc.gnu.org/onlinedocs/gcc.pdf"),
        ("gcc_internals.pdf", "https://gcc.gnu.org/onlinedocs/gccint.pdf"),
        ("basics_compiler_design.pdf", "https://www.diku.dk/~torbenm/Basics/basics_lulu2.pdf"),
    ],
    "ai_ml": [
        ("d2l_en.pdf", "https://d2l.ai/d2l-en.pdf"),
        ("sutton_barto_rl.pdf", "http://incompleteideas.net/book/RLbook2020.pdf"),
    ],
    "ostep": [
        ("cpu-intro.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-intro.pdf"),
        ("cpu-mechanisms.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-mechanisms.pdf"),
        ("cpu-sched.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-sched.pdf"),
        ("vm-intro.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/vm-intro.pdf"),
        ("vm-paging.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/vm-paging.pdf"),
        ("vm-api.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/vm-api.pdf"),
        ("vm-mechanism.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/vm-mechanism.pdf"),
        ("vm-tlbs.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/vm-tlbs.pdf"),
        ("vm-beyondphys.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/vm-beyondphys.pdf"),
        ("vm-beyondphys-policy.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/vm-beyondphys-policy.pdf"),
        ("vm-complete.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/vm-complete.pdf"),
        ("threads-intro.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/threads-intro.pdf"),
        ("threads-locks.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/threads-locks.pdf"),
        ("threads-locks-usage.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/threads-locks-usage.pdf"),
        ("threads-cv.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/threads-cv.pdf"),
        ("threads-sema.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/threads-sema.pdf"),
        ("threads-bugs.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/threads-bugs.pdf"),
        ("file-intro.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/file-intro.pdf"),
        ("file-devices.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/file-devices.pdf"),
        ("file-disks.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/file-disks.pdf"),
        ("file-implementation.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/file-implementation.pdf"),
        ("file-ffs.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/file-ffs.pdf"),
        ("file-journaling.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/file-journaling.pdf"),
        ("file-lfs.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/file-lfs.pdf"),
        ("file-integrity.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/file-integrity.pdf"),
        ("dist-intro.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/dist-intro.pdf"),
        ("dist-nfs.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/dist-nfs.pdf"),
        ("dist-afs.pdf", "https://pages.cs.wisc.edu/~remzi/OSTEP/dist-afs.pdf"),
    ],
    "networking": [
        ("rfc793_tcp.txt", "https://www.rfc-editor.org/rfc/rfc793.txt"),
        ("rfc791_ip.txt", "https://www.rfc-editor.org/rfc/rfc791.txt"),
        ("rfc2616_http.txt", "https://www.rfc-editor.org/rfc/rfc2616.txt"),
        ("rfc1035_dns.txt", "https://www.rfc-editor.org/rfc/rfc1035.txt"),
        ("rfc8446_tls13.txt", "https://www.rfc-editor.org/rfc/rfc8446.txt"),
        ("rfc5246_tls12.txt", "https://www.rfc-editor.org/rfc/rfc5246.txt"),
        ("rfc7230_http11.txt", "https://www.rfc-editor.org/rfc/rfc7230.txt"),
        ("rfc7231_http_semantics.txt", "https://www.rfc-editor.org/rfc/rfc7231.txt"),
        ("rfc4271_bgp.txt", "https://www.rfc-editor.org/rfc/rfc4271.txt"),
    ],
}

CRAWL_HTML_SOURCES = [
    {
        "category": "compilers",
        "subfolder": "crafting_interpreters",
        "index_url": "https://craftinginterpreters.com/contents.html",
        "link_pattern": r"https://craftinginterpreters\.com/[a-z0-9\-]+\.html",
    },
    {
        "category": "systems",
        "subfolder": "bottom_up_cs",
        "index_url": "https://www.bottomupcs.com/",
        "link_pattern": r"https://www\.bottomupcs\.com/(chapter\d+|index|[a-z0-9_]+)\.html",
    },
    {
        "category": "databases",
        "subfolder": "redbook",
        "index_url": "http://www.redbook.io/",
        "link_pattern": r"http://www\.redbook\.io/([a-z0-9\-]+)\.html",
    },
    {
        "category": "python_docs",
        "subfolder": "tutorial",
        "index_url": "https://docs.python.org/3/tutorial/index.html",
        "link_pattern": r"https://docs\.python\.org/3/tutorial/[a-z0-9_]+\.html",
    },
    {
        "category": "python_docs",
        "subfolder": "library",
        "index_url": "https://docs.python.org/3/library/index.html",
        "link_pattern": r"https://docs\.python\.org/3/library/(os|sys|builtins|functions|stdtypes|exceptions|collections|itertools|functools|asyncio|threading|multiprocessing|socket|re|json|math|typing)\.html",
    },
    {
        "category": "cython_docs",
        "subfolder": "",
        "index_url": "https://cython.readthedocs.io/en/latest/src/tutorial/index.html",
        "link_pattern": r"https://cython\.readthedocs\.io/en/latest/src/tutorial/[a-z0-9_]+\.html",
    },
]

LECTURE_NOTE_SOURCES = [
    {
        "category": "lecture_notes",
        "subfolder": "cmu_213",
        "index_url": "https://www.cs.cmu.edu/~213/lectures/",
        "link_pattern": r"\.pdf$",
    }
]

PWL_SUBFOLDERS = [
    "algorithms",
    "distributed_systems",
    "machine_learning",
    "operating_systems",
    "programming_languages",
    "security",
    "garbage_collection",
]

LOCAL_REQUIRED = {
    "architecture_local": [
        "patterson_hennessy_cod.pdf",
        "patterson_hennessy_caa.pdf",
    ]
}


# ==============================================================================
# 2. Yardımcı Fonksiyonlar (Dosya, Ağ, Metin Çıkarma)
# ==============================================================================

def remove_readonly(func, path, _):
    """Windows'ta git reposu silinirken salt-okunur dosyaların silinmesini sağlar."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def write_license_note(dest_dir: Path, category_name: str):
    """Kategoriye özel LICENSE_NOTE.txt dosyasını oluşturur."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    note = LICENSE_NOTES.get(category_name, "Educational and research use.")
    with open(dest_dir / "LICENSE_NOTE.txt", "w", encoding="utf-8") as f:
        f.write(note.strip() + "\n")


def format_size(bytes_num: int) -> str:
    """Okunabilir boyut formatı döndürür."""
    if bytes_num < 1024:
        return f"{bytes_num} B"
    elif bytes_num < 1024 * 1024:
        return f"{bytes_num / 1024:.1f} KB"
    else:
        return f"{bytes_num / (1024 * 1024):.2f} MB"


def download_file_stream(session: requests.Session, url: str, dest_path: Path, timeout: int = 120) -> bool:
    """Büyük dosyaları chunk'lar halinde diske indirir (Google Drive token desteğiyle)."""
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        resp = session.get(url, stream=True, timeout=timeout)

        # Google Drive büyük dosya onay sayfası kontrolü
        if "google.com" in url:
            confirm_token = None
            for key, val in resp.cookies.items():
                if key.startswith("download_warning"):
                    confirm_token = val
                    break
            if confirm_token:
                sep = "&" if "?" in url else "?"
                confirm_url = f"{url}{sep}confirm={confirm_token}"
                resp = session.get(confirm_url, stream=True, timeout=timeout)

        if resp.status_code != 200:
            logger.warning(f"HTTP {resp.status_code}: {url}")
            return False

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        # HTML hata sayfası dönüp dönmediğini kontrol et
        if dest_path.stat().st_size < 1024:
            with open(dest_path, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(200).lower()
                if "<html" in head or "404 not found" in head or "forbidden" in head:
                    dest_path.unlink()
                    return False

        return True
    except Exception as e:
        logger.warning(f"İndirme hatası ({url}): {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def extract_text_from_html(html_path: Path) -> bool:
    """HTML içeriğini temizleyip düz metin .txt olarak kaydeder."""
    txt_path = html_path.with_suffix(".txt")
    if txt_path.exists() and txt_path.stat().st_size > 0:
        return True

    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "lxml")
        for tag in soup(["script", "style", "nav", "aside", "header", "footer", "meta", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        clean_text = "\n".join(chunk for chunk in lines if chunk)

        if not clean_text:
            return False

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
        return True
    except Exception as e:
        logger.warning(f"HTML extraction hatası ({html_path.name}): {e}")
        return False


def extract_text_from_pdf(pdf_path: Path) -> bool:
    """PDF'i sayfa sayfa okuyup .txt üretir."""
    txt_path = pdf_path.with_suffix(".txt")
    if txt_path.exists() and txt_path.stat().st_size > 0:
        return True

    try:
        reader = pypdf.PdfReader(str(pdf_path))
        pages_text = []
        for page_idx, page in enumerate(reader.pages):
            try:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
            except Exception:
                continue

        if not pages_text:
            logger.warning(f"PDF'ten metin çıkarılamadı (boş veya resim): {pdf_path.name}")
            return False

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(pages_text))
        return True
    except Exception as e:
        logger.warning(f"PDF extraction hatası ({pdf_path.name}): {e}")
        return False


# ==============================================================================
# 3. İndirme ve İşleme Motoru
# ==============================================================================

def process_direct_files(session: requests.Session, skip_list: List[Dict], pdf_stats: Dict):
    """Doğrudan PDF ve TXT/HTML dosyalarını indirir ve metin çıkarır."""
    logger.info(">>> Doğrudan Dosyalar (PDF & TXT & HTML) İşleniyor...")
    for cat_name, file_list in DIRECT_PDF_SOURCES.items():
        cat_dir = BASE_DIR / cat_name
        write_license_note(cat_dir, cat_name)

        for filename, url in file_list:
            dest_file = cat_dir / filename
            ext = dest_file.suffix.lower()

            # Daha önce indirildiyse ve metni varsa tekrar indirme
            if dest_file.exists() and dest_file.stat().st_size > 1024:
                size_str = format_size(dest_file.stat().st_size)
                print(f"[OK] {cat_name}/{filename} ({size_str}) [Önbellek]")
            else:
                ok = download_file_stream(session, url, dest_file, timeout=120)
                if not ok:
                    print(f"[SKIP] {cat_name}/{url} (İndirilemedi / Timeout / 404)")
                    skip_list.append({"category": cat_name, "url": url, "reason": "Download Failed / 404"})
                    continue
                size_str = format_size(dest_file.stat().st_size)
                print(f"[OK] {cat_name}/{filename} ({size_str})")

            # Metin Çıkarma
            if ext == ".pdf":
                pdf_stats["total"] += 1
                if extract_text_from_pdf(dest_file):
                    pdf_stats["success"] += 1
            elif ext == ".html":
                extract_text_from_html(dest_file)


def process_architecture_open_access(session: requests.Session, skip_list: List[Dict], pdf_stats: Dict):
    """Açık erişimli mimari kaynaklarını (IIT Delhi ve Rowan OER) dener."""
    logger.info(">>> Açık Erişim Mimari Kaynakları (IIT Delhi & Rowan) Denetleniyor...")
    arch_dir = BASE_DIR / "architecture"
    arch_dir.mkdir(parents=True, exist_ok=True)

    # 1. IIT Delhi - Sarangi (Basic Computer Architecture)
    iit_url = "https://www.cse.iitd.ernet.in/~srsarangi/archbook/"
    try:
        resp = session.get(iit_url, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf"):
                    pdf_url = urljoin(iit_url, href)
                    fname = os.path.basename(urlparse(pdf_url).path)
                    fdest = arch_dir / f"sarangi_{fname}"
                    if not fdest.exists():
                        if download_file_stream(session, pdf_url, fdest, timeout=60):
                            print(f"[OK] architecture/{fdest.name} ({format_size(fdest.stat().st_size)})")
                            pdf_stats["total"] += 1
                            if extract_text_from_pdf(fdest):
                                pdf_stats["success"] += 1
        else:
            print(f"[SKIP] architecture/{iit_url} (HTTP {resp.status_code} Forbidden/Unavailable)")
            skip_list.append({"category": "architecture", "url": iit_url, "reason": f"HTTP {resp.status_code}"})
    except Exception as e:
        print(f"[SKIP] architecture/{iit_url} ({e})")
        skip_list.append({"category": "architecture", "url": iit_url, "reason": str(e)})

    # 2. Rowan OER - Computer Organization with ARM64 / MIPS
    rowan_url = "https://rdw.rowan.edu/cgi/viewcontent.cgi?article=1008&context=oer"
    fdest_rowan = arch_dir / "rowan_computer_organization.pdf"
    if fdest_rowan.exists() and fdest_rowan.stat().st_size > 1024:
        print(f"[OK] architecture/{fdest_rowan.name} ({format_size(fdest_rowan.stat().st_size)}) [Önbellek]")
    else:
        try:
            resp = session.get(rowan_url, timeout=20, stream=True)
            if resp.status_code == 200 and "application/pdf" in resp.headers.get("Content-Type", ""):
                with open(fdest_rowan, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                print(f"[OK] architecture/{fdest_rowan.name} ({format_size(fdest_rowan.stat().st_size)})")
                pdf_stats["total"] += 1
                if extract_text_from_pdf(fdest_rowan):
                    pdf_stats["success"] += 1
            else:
                reason = f"HTTP {resp.status_code} (Rowan Bot-Defense/Protected)"
                print(f"[SKIP] architecture/{rowan_url} ({reason})")
                skip_list.append({"category": "architecture", "url": rowan_url, "reason": reason})
        except Exception as e:
            print(f"[SKIP] architecture/{rowan_url} ({e})")
            skip_list.append({"category": "architecture", "url": rowan_url, "reason": str(e)})


def process_html_crawler(session: requests.Session, skip_list: List[Dict]):
    """HTML tabanlı açık kitapları ana sayfasından tarayıp tüm bölümleri indirir."""
    logger.info(">>> HTML Kitaplar & Dokümantasyon Taranıyor...")
    for item in CRAWL_HTML_SOURCES:
        cat_name = item["category"]
        subfolder = item["subfolder"]
        index_url = item["index_url"]
        link_pattern = re.compile(item["link_pattern"], re.IGNORECASE)

        target_dir = BASE_DIR / cat_name / subfolder if subfolder else BASE_DIR / cat_name
        write_license_note(target_dir, cat_name)

        try:
            resp = session.get(index_url, timeout=30)
            if resp.status_code != 200:
                print(f"[SKIP] {cat_name}/{index_url} (HTTP {resp.status_code})")
                skip_list.append({"category": cat_name, "url": index_url, "reason": f"HTTP {resp.status_code}"})
                continue

            soup = BeautifulSoup(resp.content, "lxml")
            discovered_links: Set[str] = set()

            for a_tag in soup.find_all("a", href=True):
                full_url = urljoin(index_url, a_tag["href"]).split("#")[0]
                if link_pattern.search(full_url):
                    discovered_links.add(full_url)

            # Ana sayfayı da ekle
            discovered_links.add(index_url)

            for page_url in sorted(discovered_links):
                parsed = urlparse(page_url)
                name = os.path.basename(parsed.path) or "index.html"
                if not name.endswith(".html"):
                    name += ".html"
                dest_file = target_dir / name

                if dest_file.exists() and dest_file.stat().st_size > 0:
                    pass
                else:
                    try:
                        p_resp = session.get(page_url, timeout=25)
                        if p_resp.status_code == 200:
                            with open(dest_file, "wb") as f:
                                f.write(p_resp.content)
                            print(f"[OK] {cat_name}/{dest_file.name} ({format_size(len(p_resp.content))})")
                        else:
                            print(f"[SKIP] {cat_name}/{page_url} (HTTP {p_resp.status_code})")
                            skip_list.append({"category": cat_name, "url": page_url, "reason": f"HTTP {p_resp.status_code}"})
                            continue
                    except Exception as e:
                        print(f"[SKIP] {cat_name}/{page_url} ({e})")
                        skip_list.append({"category": cat_name, "url": page_url, "reason": str(e)})
                        continue

                extract_text_from_html(dest_file)

        except Exception as e:
            print(f"[SKIP] {cat_name}/{index_url} (Genel Tarama Hatası: {e})")
            skip_list.append({"category": cat_name, "url": index_url, "reason": str(e)})


def process_lecture_notes(session: requests.Session, skip_list: List[Dict], pdf_stats: Dict):
    """Üniversite ders notları ve slaytlarını (CMU vb.) indirir."""
    logger.info(">>> Ders Notları & Slaytlar Taranıyor...")
    for item in LECTURE_NOTE_SOURCES:
        cat_name = item["category"]
        subfolder = item["subfolder"]
        index_url = item["index_url"]
        target_dir = BASE_DIR / cat_name / subfolder if subfolder else BASE_DIR / cat_name
        write_license_note(target_dir, cat_name)

        try:
            resp = session.get(index_url, timeout=30)
            if resp.status_code != 200:
                print(f"[SKIP] {cat_name}/{index_url} (HTTP {resp.status_code})")
                skip_list.append({"category": cat_name, "url": index_url, "reason": f"HTTP {resp.status_code}"})
                continue

            soup = BeautifulSoup(resp.content, "lxml")
            pdf_links = set()
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf"):
                    pdf_links.add(urljoin(index_url, href))

            for pdf_url in sorted(pdf_links):
                fname = os.path.basename(urlparse(pdf_url).path)
                dest_file = target_dir / fname

                if dest_file.exists() and dest_file.stat().st_size > 1024:
                    print(f"[OK] {cat_name}/{fname} ({format_size(dest_file.stat().st_size)}) [Önbellek]")
                else:
                    ok = download_file_stream(session, pdf_url, dest_file, timeout=40)
                    if not ok:
                        print(f"[SKIP] {cat_name}/{pdf_url} (İndirilemedi)")
                        skip_list.append({"category": cat_name, "url": pdf_url, "reason": "Lecture note 404/fail"})
                        continue
                    print(f"[OK] {cat_name}/{fname} ({format_size(dest_file.stat().st_size)})")

                pdf_stats["total"] += 1
                if extract_text_from_pdf(dest_file):
                    pdf_stats["success"] += 1

        except Exception as e:
            print(f"[SKIP] {cat_name}/{index_url} (Ders Notu Hatası: {e})")
            skip_list.append({"category": cat_name, "url": index_url, "reason": str(e)})


def process_papers_we_love(skip_list: List[Dict], pdf_stats: Dict):
    """Papers We Love makale koleksiyonunu yerel kontrol eder veya klonlar."""
    logger.info(">>> Papers We Love Makale Koleksiyonu Denetleniyor...")
    cat_dir = BASE_DIR / "papers_we_love"
    cat_dir.mkdir(parents=True, exist_ok=True)
    write_license_note(cat_dir, "papers_we_love")

    existing_pdfs = list(cat_dir.rglob("*.pdf"))
    if len(existing_pdfs) >= 10:
        logger.info(f"Papers We Love önbellekte mevcut ({len(existing_pdfs)} PDF). Klonlama atlandı.")
        for pdf_path in existing_pdfs:
            pdf_stats["total"] += 1
            if extract_text_from_pdf(pdf_path):
                pdf_stats["success"] += 1
        return

    repo_url = "https://github.com/papers-we-love/papers-we-love.git"
    temp_dir = tempfile.mkdtemp(prefix="pwl_clone_")

    try:
        logger.info("git clone --depth 1 başlatıldı...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=180,
        )

        repo_path = Path(temp_dir)
        all_dirs = [d for d in repo_path.iterdir() if d.is_dir()]

        for target_cat in PWL_SUBFOLDERS:
            matched = None
            norm_target = target_cat.replace("-", "_").lower()
            for d in all_dirs:
                if d.name.replace("-", "_").lower() == norm_target:
                    matched = d
                    break

            if not matched:
                logger.warning(f"PWL alt klasörü bulunamadı: {target_cat}")
                continue

            sub_dest = cat_dir / target_cat
            sub_dest.mkdir(parents=True, exist_ok=True)

            for file_path in matched.glob("**/*"):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in [".pdf", ".md", ".txt"]:
                        dest_file = sub_dest / file_path.name
                        if not dest_file.exists():
                            shutil.copy2(file_path, dest_file)
                            print(f"[OK] papers_we_love/{target_cat}/{file_path.name} ({format_size(dest_file.stat().st_size)})")

                        if ext == ".pdf":
                            pdf_stats["total"] += 1
                            if extract_text_from_pdf(dest_file):
                                pdf_stats["success"] += 1

    except Exception as e:
        msg = f"Papers We Love klonlama hatası: {e}"
        print(f"[SKIP] papers_we_love/{repo_url} ({msg})")
        skip_list.append({"category": "papers_we_love", "url": repo_url, "reason": msg})

    finally:
        shutil.rmtree(temp_dir, onerror=remove_readonly)


def check_local_required():
    """Telifli veya kullanıcı tarafından sağlanması gereken kitapları denetler."""
    logger.info(">>> Yerel Gerekli Kitaplar Denetleniyor...")
    for cat_name, files in LOCAL_REQUIRED.items():
        cat_dir = BASE_DIR / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)
        write_license_note(cat_dir, cat_name)

        for fname in files:
            fpath = cat_dir / fname
            if fpath.exists() and fpath.stat().st_size > 0:
                print(f"[OK] {cat_name}/{fname} ({format_size(fpath.stat().st_size)})")
                extract_text_from_pdf(fpath)
            else:
                print(f"[EKSİK] {cat_name}/{fname}")
                print("       Talimat: Bu telifli kitaplar otomatik indirilemez. PDF'lerini data/raw/cs_corpus/architecture_local/ altına elle koy, sonra bu script'i tekrar çalıştır.")


# ==============================================================================
# 4. Ana Çalıştırma Döngüsü ve Raporlama
# ==============================================================================

def main():
    start_time = time.time()
    logger.info("==========================================")
    logger.info("Mergen-SNN — Bilgisayar Mimarisi & CS Korpus Hattı")
    logger.info("==========================================")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    skip_list: List[Dict] = []
    pdf_stats = {"total": 0, "success": 0}

    # 1. Doğrudan Dosyalar (Patterson & Hennessy, C++, Postgres, D2L, Sutton & Barto vb.)
    process_direct_files(session, skip_list, pdf_stats)

    # 2. Açık Erişim Mimari Kitapları (IIT Delhi & Rowan)
    process_architecture_open_access(session, skip_list, pdf_stats)

    # 3. HTML tabanlı kitaplar (Crafting Interpreters, BottomUpCS, RedBook, Python Docs vb.)
    process_html_crawler(session, skip_list)

    # 4. Ders notları ve slaytlar (CMU 15-213 vb.)
    process_lecture_notes(session, skip_list, pdf_stats)

    # 5. Papers We Love makale arşivi
    process_papers_we_love(skip_list, pdf_stats)

    # 6. Yerel telifli kitap denetimi
    check_local_required()

    # ==========================================================================
    # İstatistikler ve Kategori Özeti
    # ==========================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("KATEGORİ BAZINDA CS KORPUS ÖZETİ")
    print("=" * 70)

    total_all_bytes = 0
    total_txt_bytes = 0
    total_file_count = 0
    total_txt_count = 0

    categories = sorted([d.name for d in BASE_DIR.iterdir() if d.is_dir()])
    for cat in categories:
        cat_dir = BASE_DIR / cat
        all_files = [f for f in cat_dir.rglob("*") if f.is_file()]
        txt_files = [f for f in all_files if f.suffix.lower() == ".txt" and f.name != "LICENSE_NOTE.txt"]

        cat_total_size = sum(f.stat().st_size for f in all_files)
        cat_txt_size = sum(f.stat().st_size for f in txt_files)

        total_all_bytes += cat_total_size
        total_txt_bytes += cat_txt_size
        total_file_count += len(all_files)
        total_txt_count += len(txt_files)

        print(
            f"- {cat:<20}: {len(all_files):>3} dosya ({format_size(cat_total_size):>9}) | "
            f".txt: {len(txt_files):>3} adet ({format_size(cat_txt_size):>9})"
        )

    print("-" * 70)
    print(f"Toplam Dosya Sayısı       : {total_file_count}")
    print(f"Toplam Çıkarılan .txt     : {total_txt_count}")
    print(f"Toplam Ham + TXT Boyutu   : {format_size(total_all_bytes)}")
    print(f"TOPLAM ÇIKARILAN METİN    : {format_size(total_txt_bytes)} ({total_txt_bytes / (1024 * 1024):.2f} MB)")
    hedef_durumu = "[BASARILI - HEDEF ASILDI]" if total_txt_bytes >= 50 * 1024 * 1024 else "[50 MB ALTINDA]"
    print(f"Hedef Kontrolü (>= 50 MB) : {hedef_durumu}")
    print(f"Toplam Hata / SKIP        : {len(skip_list)}")
    if pdf_stats["total"] > 0:
        pct = (pdf_stats["success"] / pdf_stats["total"]) * 100
        print(f"PDF Extraction Başarısı   : {pdf_stats['success']}/{pdf_stats['total']} (%{pct:.1f})")
    print(f"Toplam Çalışma Süresi     : {elapsed:.1f} saniye")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

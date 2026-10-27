import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import pypdf
import requests

from mergen.utils.logger import get_logger

logger = get_logger("mergen.fetch_corpus")

BASE_DIR = Path("data/raw/cs_corpus")
RETRY_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

# ==============================================================================
# 1. Lisans Notları ve Kaynak Tanımları
# ==============================================================================

LICENSE_NOTES = {
    "architecture": "Computer Architecture reference textbooks and open ISA specifications (Patterson & Hennessy, RISC-V International, ARM Developer, Sarangi).",
    "algorithms": "Open educational resources and open-access algorithmic textbooks and lecture notes (Jeff Erickson, AlgoXY, OpenDataStructures, MIT 6.006).",
    "theory": "Open textbooks in theoretical computer science and discrete mathematics (Critchlow & Eck, MIT OCW, ArXiv).",
    "systems": "Open systems textbooks and reference materials (Max Hailperin OS, Dive into Systems, Bottom Up CS, Pro Git).",
    "compilers": "Open compiler textbooks and reference manuals (Crafting Interpreters, GCC Manual, TutorialsPoint).",
    "databases": "PostgreSQL Reference Documentation and Readings in Database Systems (The Red Book 5th ed.).",
    "networking": "Open Computer Networking principles and foundational IETF RFC standards (IETF / computer-networking.info).",
    "ai_ml": "Open deep learning and reinforcement learning books (Dive into Deep Learning - D2L, Sutton & Barto RL).",
    "papers_we_love": "Academic CS papers curated by the Papers We Love community, hosted for educational research.",
    "lecture_notes": "Open courseware and university lecture slides (MIT 6.006 Algorithms, CMU, MIT OCW).",
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
        # Patterson & Hennessy Kitapları
        ("patterson_hennessy_cod_5th.pdf", "https://docs.google.com/uc?export=download&id=1OwVqBQkyjW4xvUvrVYyx3zMe5XkRHRlX"),
        ("patterson_hennessy_caa_6th.pdf", "https://archive.org/download/computerarchitectureaquantitativeapproach6thedition/Computer%20Architecture%3A%20A%20Quantitative%20Approach%206th%20Edition.pdf"),
        ("patterson_hennessy_caa_4th.pdf", "https://github.com/lovisticsdev/engineering_books/raw/main/Computer%20Architecture/Hennessy%2C%20Patterson%20-%20Computer%20Architecture%20A%20Quantitative%20Approach%20%5B4th%20Edition%5D.pdf"),
        ("patterson_hennessy_cod_riscv.pdf", "https://github.com/kk-1024b/CS-textbooks/raw/main/Computer-Organization-and-Design-The-Hardware-Software-Interface-RISC-V-Edition.pdf"),
        # Açık Erişim Mimari ve ISA Spesifikasyonları
        ("riscv_unprivileged_spec.pdf", "https://github.com/riscv/riscv-isa-manual/releases/download/Ratified-IMAFDQC/riscv-spec-20191213.pdf"),
        ("riscv_privileged_spec.pdf", "https://github.com/riscv/riscv-isa-manual/releases/download/Priv-v1.12/riscv-privileged-20211203.pdf"),
        ("xv6_riscv_architecture_book.pdf", "https://pdos.csail.mit.edu/6.828/2023/xv6/book-riscv-rev3.pdf"),
        ("armv8a_architecture_manual.pdf", "https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Learn%20the%20Architecture/Armv8-A%20Instruction%20Set%20Architecture.pdf"),
        ("sarangi_adv_architecture.pdf", "https://srsarangi.github.io/advbook/nextgen-comp-arch.pdf"),
    ],
    "theory": [
        ("foundations_of_computation.pdf", "https://math.hws.edu/FoundationsOfComputation/FoundationsOfComputation_2.3.2_8.5x11.pdf"),
        ("mit_mcs.pdf", "https://courses.csail.mit.edu/6.042/spring18/mcs.pdf"),
        ("theory_of_computing_arxiv.pdf", "https://arxiv.org/pdf/2010.10239.pdf"),
        ("theory_of_computing_stfx.pdf", "https://people.stfx.ca/tasmith/theory/TheoryOfComputing.pdf"),
    ],
    "compilers": [
        ("gcc_manual.pdf", "https://gcc.gnu.org/onlinedocs/gcc.pdf"),
        ("gcc_internals.pdf", "https://gcc.gnu.org/onlinedocs/gccint.pdf"),
        ("tutorialspoint_compiler_design.pdf", "https://www.tutorialspoint.com/compiler_design/compiler_design_tutorial.pdf"),
        ("basics_compiler_design.pdf", "https://www.diku.dk/~torbenm/Basics/basics_lulu2.pdf"),
    ],
    "systems": [
        ("progit_2nd_edition.pdf", "https://github.com/progit/progit2/releases/download/2.1.423/progit.pdf"),
        ("gdb_debugging_manual.pdf", "https://sourceware.org/gdb/current/onlinedocs/gdb.pdf"),
        ("sarangi_os_book.pdf", "https://srsarangi.github.io/osbook/osbook-v0.96.pdf"),
        ("os_book_max.pdf", "https://gustavus.edu/mcs/max/os-book/os-book.pdf"),
        ("dive_into_systems.pdf", "https://diveintosystems.org/book/pdf/DiveIntoSystems.pdf"),
    ],
    "cpp_docs": [
        ("cpp23_standard_draft_n4950.pdf", "https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/n4950.pdf"),
        ("cpp20_standard_draft_n4861.pdf", "https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/n4861.pdf"),
        ("cpp17_standard_draft_n4659.pdf", "https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/n4659.pdf"),
    ],
    "databases": [
        ("postgresql_16_documentation.pdf", "https://www.postgresql.org/files/documentation/pdf/16/postgresql-16-A4.pdf"),
        ("sqlite_arch.html", "https://www.sqlite.org/arch.html"),
        ("sqlite_opcode.html", "https://www.sqlite.org/opcode.html"),
        ("sqlite_fileformat.html", "https://www.sqlite.org/fileformat2.html"),
    ],
    "c_docs": [
        ("c17_standard_n2310.pdf", "https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2310.pdf"),
        ("c11_standard_n1570.pdf", "https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf"),
        ("c99_standard_n1256.pdf", "https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1256.pdf"),
    ],
    "algorithms": [
        ("algorithms_jeffe.pdf", "https://jeffe.cs.illinois.edu/teaching/algorithms/book/Algorithms-JeffE.pdf"),
        ("elementary_algorithms.pdf", "https://github.com/liuxinyu95/AlgoXY/releases/download/v0.618033/elementary-algorithms.pdf"),
        ("ods_python.pdf", "https://opendatastructures.org/ods-python.pdf"),
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
        "link_pattern": r"https://docs\.python\.org/3/library/(os|sys|builtins|functions|stdtypes|exceptions|collections|itertools|functools|asyncio|threading|multiprocessing|socket|re|json|math|typing|ctypes|dis|inspect|ast|subprocess|time|datetime|io|struct|gc|tracemalloc|mmap|signal|shutil)\.html",
    },
    {
        "category": "cython_docs",
        "subfolder": "",
        "index_url": "https://cython.readthedocs.io/en/latest/src/tutorial/index.html",
        "link_pattern": r"https://cython\.readthedocs\.io/en/latest/src/tutorial/[a-z0-9_]+\.html",
    },
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
# 2. Yardımcı Fonksiyonlar (Dosya, Ağ, Metin Çıkarma, Gelişmiş İndirici)
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


def try_download_stream(session: requests.Session, url: str, dest_path: Path, timeout: int = 120) -> bool:
    """Tek bir URL'yi akış (stream) yöntemiyle indirmeyi dener."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    resp = session.get(url, stream=True, timeout=timeout, headers=RETRY_HEADERS)

    # Google Drive onay sayfası kontrolü
    if "google.com" in url:
        confirm_token = None
        for key, val in resp.cookies.items():
            if key.startswith("download_warning"):
                confirm_token = val
                break
        if confirm_token:
            sep = "&" if "?" in url else "?"
            confirm_url = f"{url}{sep}confirm={confirm_token}"
            resp = session.get(confirm_url, stream=True, timeout=timeout, headers=RETRY_HEADERS)

    if resp.status_code != 200:
        return False

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    # Küçük HTML hata sayfası dönüp dönmediğini kontrol et
    if dest_path.stat().st_size < 1024:
        with open(dest_path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(200).lower()
            if "<html" in head or "404 not found" in head or "forbidden" in head:
                dest_path.unlink()
                return False

    return True


def smart_download(session: requests.Session, url: str, dest_path: Path, timeout: int = 120) -> Tuple[bool, str]:
    """Çok aşamalı indirme motoru: Orijinal -> HTTPS/HTTP Fallback -> Wayback Fallback."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Aşama: Orijinal URL
    try:
        if try_download_stream(session, url, dest_path, timeout=timeout):
            return True, url
    except Exception as e:
        logger.debug(f"Orijinal deneme hatası ({url}): {e}")

    # 2. Aşama: HTTPS -> HTTP Fallback
    if url.startswith("https://"):
        http_url = url.replace("https://", "http://", 1)
        try:
            if try_download_stream(session, http_url, dest_path, timeout=timeout):
                return True, http_url
        except Exception:
            pass

    # 3. Aşama: Wayback Machine Fallback
    wayback_url = f"https://web.archive.org/web/2024/{url}"
    try:
        if try_download_stream(session, wayback_url, dest_path, timeout=timeout):
            return True, wayback_url
    except Exception:
        pass

    return False, url


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
    """Doğrudan PDF ve TXT/HTML dosyalarını akıllı indirme motoruyla indirir."""
    logger.info(">>> Doğrudan Dosyalar (PDF & TXT & HTML) Akıllı İndiriciyle İşleniyor...")
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
                ok, final_url = smart_download(session, url, dest_file, timeout=120)
                if not ok:
                    print(f"[SKIP] {cat_name}/{url} (İndirilemedi / Timeout / 404 / Wayback Başarısız)")
                    skip_list.append({"category": cat_name, "url": url, "reason": "Download Failed / 404 / Archive Fail"})
                    continue
                size_str = format_size(dest_file.stat().st_size)
                via_str = f" via {final_url}" if final_url != url else ""
                print(f"[OK] {cat_name}/{filename} ({size_str}){via_str}")

            # Metin Çıkarma
            if ext == ".pdf":
                pdf_stats["total"] += 1
                if extract_text_from_pdf(dest_file):
                    pdf_stats["success"] += 1
            elif ext == ".html":
                extract_text_from_html(dest_file)


def process_hailperin_os(pdf_stats: Dict):
    """Max Hailperin OS kitabını resmi GitHub reposundan çekip sistem kategorisine ekler."""
    logger.info(">>> Hailperin OS GitHub Reposu İşleniyor...")
    sys_dir = BASE_DIR / "systems"
    sys_dir.mkdir(parents=True, exist_ok=True)
    dest_file = sys_dir / "hailperin_os_book.pdf"

    if dest_file.exists() and dest_file.stat().st_size > 1024:
        print(f"[OK] systems/{dest_file.name} ({format_size(dest_file.stat().st_size)}) [Önbellek]")
        pdf_stats["total"] += 1
        if extract_text_from_pdf(dest_file):
            pdf_stats["success"] += 1
        return

    repo_url = "https://github.com/Max-Hailperin/Operating-Systems-and-Middleware--Supporting-Controlled-Interaction.git"
    temp_dir = tempfile.mkdtemp(prefix="hailperin_")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        # Bulunan en büyük kitap PDF'ini seç
        candidates = []
        for root, _, files in os.walk(temp_dir):
            for f in files:
                if f.lower().endswith(".pdf") and ("osm" in f.lower() or "free-osbook" in f.lower()):
                    full_p = Path(root) / f
                    candidates.append((full_p.stat().st_size, full_p))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            chosen_pdf = candidates[0][1]
            shutil.copy2(chosen_pdf, dest_file)
            print(f"[OK] systems/{dest_file.name} ({format_size(dest_file.stat().st_size)}) [GitHub Clone]")
            pdf_stats["total"] += 1
            if extract_text_from_pdf(dest_file):
                pdf_stats["success"] += 1
        else:
            print(f"[SKIP] systems/{repo_url} (Repo içinde kitap PDF'i bulunamadı)")
    except Exception as e:
        print(f"[SKIP] systems/{repo_url} (Git clone hatası: {e})")
    finally:
        shutil.rmtree(temp_dir, onerror=remove_readonly)


def process_mit_6006_lectures(session: requests.Session, pdf_stats: Dict):
    """MIT 6.006 Algoritmalar ders notlarını (lec01 - lec24) indirir."""
    logger.info(">>> MIT 6.006 Algoritmalar Ders Notları (lec01 - lec24) İndiriliyor...")
    lec_dir = BASE_DIR / "lecture_notes" / "mit_6006"
    lec_dir.mkdir(parents=True, exist_ok=True)
    write_license_note(lec_dir, "lecture_notes")

    success_count = 0
    for i in range(1, 25):
        fname = f"lec{i:02d}.pdf"
        url = f"https://courses.csail.mit.edu/6.006/spring11/lectures/{fname}"
        dest_file = lec_dir / fname

        if dest_file.exists() and dest_file.stat().st_size > 1024:
            print(f"[OK] lecture_notes/mit_6006/{fname} ({format_size(dest_file.stat().st_size)}) [Önbellek]")
            pdf_stats["total"] += 1
            if extract_text_from_pdf(dest_file):
                pdf_stats["success"] += 1
            success_count += 1
        else:
            ok, final_url = smart_download(session, url, dest_file, timeout=40)
            if ok:
                print(f"[OK] lecture_notes/mit_6006/{fname} ({format_size(dest_file.stat().st_size)})")
                pdf_stats["total"] += 1
                if extract_text_from_pdf(dest_file):
                    pdf_stats["success"] += 1
                success_count += 1
            else:
                print(f"[SKIP] lecture_notes/mit_6006/{fname} (İndirilemedi)")

    logger.info(f"MIT 6.006: {success_count}/24 ders slaytı başarıyla işlendi.")


def process_architecture_open_access(session: requests.Session, skip_list: List[Dict], pdf_stats: Dict):
    """Açık erişimli mimari kaynaklarını (IIT Delhi ve Rowan OER) dener."""
    logger.info(">>> Açık Erişim Mimari Kaynakları (IIT Delhi & Rowan) Denetleniyor...")
    arch_dir = BASE_DIR / "architecture"
    arch_dir.mkdir(parents=True, exist_ok=True)

    # 1. IIT Delhi - Sarangi (Basic Computer Architecture)
    iit_url = "https://www.cse.iitd.ernet.in/~srsarangi/archbook/"
    try:
        resp = session.get(iit_url, timeout=15, headers=RETRY_HEADERS)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf"):
                    pdf_url = urljoin(iit_url, href)
                    fname = os.path.basename(urlparse(pdf_url).path)
                    fdest = arch_dir / f"sarangi_{fname}"
                    if not fdest.exists():
                        if try_download_stream(session, pdf_url, fdest, timeout=60):
                            print(f"[OK] architecture/{fdest.name} ({format_size(fdest.stat().st_size)})")
                            pdf_stats["total"] += 1
                            if extract_text_from_pdf(fdest):
                                pdf_stats["success"] += 1
        else:
            # Alternatif GitHub Pages kopyası zaten DIRECT_PDF_SOURCES'da mevcut (sarangi_adv_architecture.pdf)
            reason = f"HTTP {resp.status_code} Forbidden (Sarangi GitHub ayna kitabı kullanıldı)"
            print(f"[SKIP] architecture/{iit_url} ({reason})")
            skip_list.append({"category": "architecture", "url": iit_url, "reason": reason})
    except Exception as e:
        print(f"[SKIP] architecture/{iit_url} ({e})")
        skip_list.append({"category": "architecture", "url": iit_url, "reason": str(e)})

    # 2. Rowan OER - Computer Organization
    rowan_url = "https://rdw.rowan.edu/cgi/viewcontent.cgi?article=1008&context=oer"
    fdest_rowan = arch_dir / "rowan_computer_organization.pdf"
    if fdest_rowan.exists() and fdest_rowan.stat().st_size > 1024:
        print(f"[OK] architecture/{fdest_rowan.name} ({format_size(fdest_rowan.stat().st_size)}) [Önbellek]")
    else:
        try:
            resp = session.get(rowan_url, timeout=20, stream=True, headers=RETRY_HEADERS)
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
                # ARM alternatifi zaten DIRECT_PDF_SOURCES'ta eklendi (armv8a_architecture_manual.pdf)
                reason = f"HTTP {resp.status_code} (Rowan Bot-Defense; ARMv8-A resmi kılavuzu kullanıldı)"
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
            resp = session.get(index_url, timeout=30, headers=RETRY_HEADERS)
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
                        p_resp = session.get(page_url, timeout=25, headers=RETRY_HEADERS)
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
    logger.info("Mergen-SNN — Dayanıklı CS Korpus Hattı")
    logger.info("==========================================")

    session = requests.Session()
    session.headers.update(RETRY_HEADERS)

    skip_list: List[Dict] = []
    pdf_stats = {"total": 0, "success": 0}

    # 1. Doğrudan Dosyalar (Patterson & Hennessy, C++, Postgres, D2L, Sutton & Barto, RFC'ler vb.)
    process_direct_files(session, skip_list, pdf_stats)

    # 2. Hailperin OS Kitabı (GitHub Clone -> systems/)
    process_hailperin_os(pdf_stats)

    # 3. MIT 6.006 Algoritmalar Ders Notları (lec01 - lec24 -> lecture_notes/)
    process_mit_6006_lectures(session, pdf_stats)

    # 4. Açık Erişim Mimari Kitapları (IIT Delhi & Rowan)
    process_architecture_open_access(session, skip_list, pdf_stats)

    # 5. HTML tabanlı kitaplar (Crafting Interpreters, BottomUpCS, RedBook, Python Docs vb.)
    process_html_crawler(session, skip_list)

    # 6. Papers We Love makale arşivi
    process_papers_we_love(skip_list, pdf_stats)

    # 7. Yerel telifli kitap denetimi
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
    hedef_durumu = "[BASARILI - HEDEF ASILDI]" if total_txt_bytes >= 65 * 1024 * 1024 else "[65 MB ALTINDA]"
    print(f"Hedef Kontrolü (>= 65 MB) : {hedef_durumu}")
    print(f"Kalan SKIP / Hata Sayısı  : {len(skip_list)}")
    if pdf_stats["total"] > 0:
        pct = (pdf_stats["success"] / pdf_stats["total"]) * 100
        print(f"PDF Extraction Başarısı   : {pdf_stats['success']}/{pdf_stats['total']} (%{pct:.1f})")
    print(f"Toplam Çalışma Süresi     : {elapsed:.1f} saniye")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

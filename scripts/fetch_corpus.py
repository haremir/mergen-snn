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
    "gc_papers": "Curated academic papers on automatic memory management and garbage collection (PWL / open access).",
    "architecture_local": "Copyrighted textbooks (Patterson & Hennessy). Local copy provided by user for personal research.",
    "tr_docs": "Turkish Computer Science articles from Wikipedia (CC-BY-SA 3.0/4.0), open-access university theses from ITU Polen / DSpace, and Turkish CS lecture notes.",
    "code_docs": "Permissively licensed Python and C source code, algorithms, and data structures (MIT/Apache/BSD) from TheAlgorithms and public code datasets.",
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
        ("arora_barak_complexity.pdf", "https://theory.cs.princeton.edu/complexity/book.pdf"),
        ("smid_theory_of_computation.pdf", "https://cglab.ca/~michiel/TheoryOfComputation/TheoryOfComputation.pdf"),
        ("savage_models_of_computation.pdf", "https://cs.brown.edu/people/jsavage/book/pdfs/ModelsOfComputation.pdf"),
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
        ("cpp11_standard_draft_n3337.pdf", "https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3337.pdf"),
    ],
    "databases": [
        ("postgresql_16_documentation.pdf", "https://www.postgresql.org/files/documentation/pdf/16/postgresql-16-A4.pdf"),
        ("sqlite_arch.html", "https://www.sqlite.org/arch.html"),
        ("sqlite_opcode.html", "https://www.sqlite.org/opcode.html"),
        ("sqlite_fileformat.html", "https://www.sqlite.org/fileformat2.html"),
    ],
    "c_docs": [
        ("c23_standard_draft_n3096.pdf", "https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3096.pdf"),
        ("c17_standard_n2310.pdf", "https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2310.pdf"),
        ("c11_standard_n1570.pdf", "https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf"),
        ("c99_standard_n1256.pdf", "https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1256.pdf"),
    ],
    "gc_papers": [
        ("ungar_generation_scavenging.pdf", "https://people.cs.umass.edu/~emery/classes/cmpsci691s-fall2004/papers/p157-ungar.pdf"),
        ("immix_pldi_2008.pdf", "https://users.cecs.anu.edu.au/~steveb/pubs/papers/immix-pldi-2008.pdf"),
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
        ("rfc768_udp.txt", "https://www.rfc-editor.org/rfc/rfc768.txt"),
        ("rfc791_ip.txt", "https://www.rfc-editor.org/rfc/rfc791.txt"),
        ("rfc8200_ipv6.txt", "https://www.rfc-editor.org/rfc/rfc8200.txt"),
        ("rfc1034_dns_concepts.txt", "https://www.rfc-editor.org/rfc/rfc1034.txt"),
        ("rfc1035_dns.txt", "https://www.rfc-editor.org/rfc/rfc1035.txt"),
        ("rfc2616_http.txt", "https://www.rfc-editor.org/rfc/rfc2616.txt"),
        ("rfc7230_http11.txt", "https://www.rfc-editor.org/rfc/rfc7230.txt"),
        ("rfc7231_http_semantics.txt", "https://www.rfc-editor.org/rfc/rfc7231.txt"),
        ("rfc7232_http_conditional.txt", "https://www.rfc-editor.org/rfc/rfc7232.txt"),
        ("rfc7233_http_range.txt", "https://www.rfc-editor.org/rfc/rfc7233.txt"),
        ("rfc7234_http_caching.txt", "https://www.rfc-editor.org/rfc/rfc7234.txt"),
        ("rfc7235_http_auth.txt", "https://www.rfc-editor.org/rfc/rfc7235.txt"),
        ("rfc9110_http_semantics.txt", "https://www.rfc-editor.org/rfc/rfc9110.txt"),
        ("rfc9111_http_caching.txt", "https://www.rfc-editor.org/rfc/rfc9111.txt"),
        ("rfc9112_http11.txt", "https://www.rfc-editor.org/rfc/rfc9112.txt"),
        ("rfc9113_http2.txt", "https://www.rfc-editor.org/rfc/rfc9113.txt"),
        ("rfc9114_http3.txt", "https://www.rfc-editor.org/rfc/rfc9114.txt"),
        ("rfc9000_quic.txt", "https://www.rfc-editor.org/rfc/rfc9000.txt"),
        ("rfc2328_ospfv2.txt", "https://www.rfc-editor.org/rfc/rfc2328.txt"),
        ("rfc2131_dhcp.txt", "https://www.rfc-editor.org/rfc/rfc2131.txt"),
        ("rfc6455_websocket.txt", "https://www.rfc-editor.org/rfc/rfc6455.txt"),
        ("rfc792_icmp.txt", "https://www.rfc-editor.org/rfc/rfc792.txt"),
        ("rfc4253_ssh_transport.txt", "https://www.rfc-editor.org/rfc/rfc4253.txt"),
        ("rfc8446_tls13.txt", "https://www.rfc-editor.org/rfc/rfc8446.txt"),
        ("rfc5246_tls12.txt", "https://www.rfc-editor.org/rfc/rfc5246.txt"),
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


def check_pdf_quality(pdf_path: Path, low_quality_list: List[Dict]):
    """Ham PDF boyutu ile çıkarılan TXT boyutunu oranlar. Oran < %2 ise konsola yazar ve listeye ekler."""
    txt_path = pdf_path.with_suffix(".txt")
    if not pdf_path.exists() or not txt_path.exists():
        return
    raw_size = pdf_path.stat().st_size
    txt_size = txt_path.stat().st_size
    if raw_size <= 0:
        return
    ratio = (txt_size / raw_size) * 100.0
    if ratio < 2.0:
        try:
            rel_path = str(pdf_path.relative_to(BASE_DIR)).replace("\\", "/")
        except Exception:
            rel_path = pdf_path.name
        print(f"[DÜŞÜK KALİTE] {rel_path}: ham={format_size(raw_size)}, txt={format_size(txt_size)} (oran: %{ratio:.2f})")
        low_quality_list.append({
            "path": rel_path,
            "raw_size": raw_size,
            "txt_size": txt_size,
            "ratio": ratio
        })


def save_low_quality_report(low_quality_list: List[Dict]):
    """Düşük kaliteli PDF listesini data/raw/cs_corpus/_low_quality.txt dosyasına kaydeder."""
    out_path = BASE_DIR / "_low_quality.txt"
    seen = set()
    unique_items = []
    for item in low_quality_list:
        if item["path"] not in seen:
            seen.add(item["path"])
            unique_items.append(item)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Mergen-SNN Korpus Kalite Denetimi: Metin Oranı < %2 Olan PDF Kaynakları\n")
        f.write("# Format: {dosya_yolu} | Ham Boyut | TXT Boyut | Metin Oranı\n\n")
        for item in sorted(unique_items, key=lambda x: x["ratio"]):
            f.write(f"{item['path']} | ham={format_size(item['raw_size'])} | txt={format_size(item['txt_size'])} | oran=%{item['ratio']:.2f}\n")
    logger.info(f"Düşük kaliteli PDF listesi yazıldı: {out_path} ({len(unique_items)} adet)")


# ==============================================================================
# 3. İndirme ve İşleme Motoru
# ==============================================================================

def process_direct_files(session: requests.Session, skip_list: List[Dict], pdf_stats: Dict, low_quality_list: List[Dict]):
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
                check_pdf_quality(dest_file, low_quality_list)
            elif ext == ".html":
                extract_text_from_html(dest_file)


def process_hailperin_os(pdf_stats: Dict, low_quality_list: List[Dict]):
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
        check_pdf_quality(dest_file, low_quality_list)
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
            check_pdf_quality(dest_file, low_quality_list)
        else:
            print(f"[SKIP] systems/{repo_url} (Repo içinde kitap PDF'i bulunamadı)")
    except Exception as e:
        print(f"[SKIP] systems/{repo_url} (Git clone hatası: {e})")
    finally:
        shutil.rmtree(temp_dir, onerror=remove_readonly)


def process_mit_6006_lectures(session: requests.Session, pdf_stats: Dict, low_quality_list: List[Dict]):
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
            check_pdf_quality(dest_file, low_quality_list)
        else:
            ok, final_url = smart_download(session, url, dest_file, timeout=40)
            if ok:
                print(f"[OK] lecture_notes/mit_6006/{fname} ({format_size(dest_file.stat().st_size)})")
                pdf_stats["total"] += 1
                if extract_text_from_pdf(dest_file):
                    pdf_stats["success"] += 1
                success_count += 1
                check_pdf_quality(dest_file, low_quality_list)
            else:
                print(f"[SKIP] lecture_notes/mit_6006/{fname} (İndirilemedi)")

    logger.info(f"MIT 6.006: {success_count}/24 ders slaytı başarıyla işlendi.")


def process_architecture_open_access(session: requests.Session, skip_list: List[Dict], pdf_stats: Dict, low_quality_list: List[Dict]):
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
                            check_pdf_quality(fdest, low_quality_list)
        else:
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
        check_pdf_quality(fdest_rowan, low_quality_list)
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
                check_pdf_quality(fdest_rowan, low_quality_list)
            else:
                reason = f"HTTP {resp.status_code} (Rowan Bot-Defense; ARMv8-A resmi kılavuzu kullanıldı)"
                print(f"[SKIP] architecture/{rowan_url} ({reason})")
                skip_list.append({"category": "architecture", "url": rowan_url, "reason": reason})
        except Exception as e:
            print(f"[SKIP] architecture/{rowan_url} ({e})")
            skip_list.append({"category": "architecture", "url": rowan_url, "reason": str(e)})


def process_gc_papers(pdf_stats: Dict, low_quality_list: List[Dict]):
    """SORUN 1: Papers We Love reposundaki garbage_collection klasörünü gc_papers'a aktarır."""
    logger.info(">>> Garbage Collection (gc_papers) Makaleleri İşleniyor...")
    gc_dir = BASE_DIR / "gc_papers"
    gc_dir.mkdir(parents=True, exist_ok=True)
    write_license_note(gc_dir, "gc_papers")

    # 1. Önce yerel papers_we_love/garbage_collection/ klasöründen kopyala (varsa)
    pwl_gc = BASE_DIR / "papers_we_love" / "garbage_collection"
    if pwl_gc.exists():
        for f in pwl_gc.glob("*.pdf"):
            dest = gc_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)

    # 2. Hala PDF yoksa veya azsa GitHub'dan klonla
    existing_pdfs = list(gc_dir.glob("*.pdf"))
    if len(existing_pdfs) < 3:
        temp_dir = tempfile.mkdtemp(prefix="pwl_gc_")
        try:
            repo_url = "https://github.com/papers-we-love/papers-we-love.git"
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, temp_dir],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=180,
            )
            src_gc = Path(temp_dir) / "garbage_collection"
            if src_gc.exists():
                for f in src_gc.glob("*.pdf"):
                    dest = gc_dir / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)
            else:
                # ls-tree ile ara
                res = subprocess.run(["git", "-C", temp_dir, "ls-tree", "-r", "HEAD", "--name-only"], capture_output=True, text=True)
                for line in res.stdout.splitlines():
                    if "garbage" in line.lower() and line.lower().endswith(".pdf"):
                        src_f = Path(temp_dir) / line
                        dest = gc_dir / src_f.name
                        if not dest.exists():
                            shutil.copy2(src_f, dest)
        except Exception as e:
            logger.warning(f"PWL gc_papers klonlama hatası: {e}")
        finally:
            shutil.rmtree(temp_dir, onerror=remove_readonly)

    # 3. Metin çıkarma ve kalite denetimi
    for pdf_file in sorted(gc_dir.glob("*.pdf")):
        pdf_stats["total"] += 1
        txt_path = pdf_file.with_suffix(".txt")
        if txt_path.exists() and txt_path.stat().st_size > 0:
            print(f"[OK] gc_papers/{pdf_file.name} ({format_size(pdf_file.stat().st_size)}) [Önbellek]")
            pdf_stats["success"] += 1
        else:
            if extract_text_from_pdf(pdf_file):
                pdf_stats["success"] += 1
                print(f"[OK] gc_papers/{pdf_file.name} ({format_size(pdf_file.stat().st_size)})")
            else:
                print(f"[UYARI] gc_papers/{pdf_file.name} metin çıkarılamadı")
        check_pdf_quality(pdf_file, low_quality_list)


def process_lecture_notes_expansion(session: requests.Session, skip_list: List[Dict], pdf_stats: Dict, low_quality_list: List[Dict]):
    """SORUN 2: Kleinberg-Tardos, Stanford CS161 ve Roughgarden ders notlarını indirir."""
    logger.info(">>> Metin Tabanlı Ders Notları (Kleinberg-Tardos, Stanford CS161, Roughgarden) İşleniyor...")

    # 1. Kleinberg & Tardos - Algorithm Design Ders Notları
    kt_dir = BASE_DIR / "lecture_notes" / "kleinberg_tardos"
    kt_dir.mkdir(parents=True, exist_ok=True)
    kt_url = "https://www.cs.princeton.edu/~wayne/kleinberg-tardos/"
    try:
        r = session.get(kt_url, timeout=20, headers=RETRY_HEADERS)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "lxml")
            pdf_links = set()
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf") and "pdf/" in href.lower():
                    pdf_links.add(urljoin(kt_url, href))

            for purl in sorted(pdf_links):
                fname = os.path.basename(urlparse(purl).path)
                dest = kt_dir / fname
                if dest.exists() and dest.stat().st_size > 1024:
                    print(f"[OK] lecture_notes/kleinberg_tardos/{fname} ({format_size(dest.stat().st_size)}) [Önbellek]")
                    pdf_stats["total"] += 1
                    if extract_text_from_pdf(dest):
                        pdf_stats["success"] += 1
                else:
                    ok, _ = smart_download(session, purl, dest, timeout=40)
                    if ok:
                        print(f"[OK] lecture_notes/kleinberg_tardos/{fname} ({format_size(dest.stat().st_size)})")
                        pdf_stats["total"] += 1
                        if extract_text_from_pdf(dest):
                            pdf_stats["success"] += 1
                    else:
                        print(f"[SKIP] lecture_notes/kleinberg_tardos/{fname} (İndirilemedi)")
                        skip_list.append({"category": "lecture_notes", "url": purl, "reason": "KT 404/fail"})
                check_pdf_quality(dest, low_quality_list)
        else:
            print(f"[SKIP] lecture_notes/{kt_url} (HTTP {r.status_code})")
            skip_list.append({"category": "lecture_notes", "url": kt_url, "reason": f"HTTP {r.status_code}"})
    except Exception as e:
        print(f"[SKIP] lecture_notes/{kt_url} ({e})")
        skip_list.append({"category": "lecture_notes", "url": kt_url, "reason": str(e)})

    # 2. Stanford CS161 - Design and Analysis of Algorithms
    st_dir = BASE_DIR / "lecture_notes" / "stanford_cs161"
    st_dir.mkdir(parents=True, exist_ok=True)
    st_url = "https://web.stanford.edu/class/archive/cs/cs161/cs161.1168/"
    try:
        r = session.get(st_url, timeout=20, headers=RETRY_HEADERS)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "lxml")
            s_links = set()
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf") and "lecture" in href.lower():
                    s_links.add(urljoin(st_url, href))

            for purl in sorted(s_links):
                fname = os.path.basename(urlparse(purl).path)
                dest = st_dir / fname
                if dest.exists() and dest.stat().st_size > 1024:
                    print(f"[OK] lecture_notes/stanford_cs161/{fname} ({format_size(dest.stat().st_size)}) [Önbellek]")
                    pdf_stats["total"] += 1
                    if extract_text_from_pdf(dest):
                        pdf_stats["success"] += 1
                else:
                    ok, _ = smart_download(session, purl, dest, timeout=40)
                    if ok:
                        print(f"[OK] lecture_notes/stanford_cs161/{fname} ({format_size(dest.stat().st_size)})")
                        pdf_stats["total"] += 1
                        if extract_text_from_pdf(dest):
                            pdf_stats["success"] += 1
                    else:
                        print(f"[SKIP] lecture_notes/stanford_cs161/{fname} (İndirilemedi)")
                        skip_list.append({"category": "lecture_notes", "url": purl, "reason": "Stanford CS161 fail"})
                check_pdf_quality(dest, low_quality_list)
        else:
            print(f"[SKIP] lecture_notes/{st_url} (HTTP {r.status_code})")
            skip_list.append({"category": "lecture_notes", "url": st_url, "reason": f"HTTP {r.status_code}"})
    except Exception as e:
        print(f"[SKIP] lecture_notes/{st_url} ({e})")
        skip_list.append({"category": "lecture_notes", "url": st_url, "reason": str(e)})

    # 3. Tim Roughgarden - Algoritmik Ders Notları (tüm açık ders notları)
    rg_dir = BASE_DIR / "lecture_notes" / "roughgarden"
    rg_dir.mkdir(parents=True, exist_ok=True)
    rg_url = "https://timroughgarden.org/notes.html"
    try:
        r = session.get(rg_url, timeout=20, headers=RETRY_HEADERS)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "lxml")
            rg_links = set()
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf") and ("/l/" in href.lower() or any(k in href.lower() for k in ["fob", "cs168", "cs261", "cs264", "wca", "algo"])):
                    rg_links.add(urljoin(rg_url, href))

            for purl in sorted(rg_links):
                parsed = urlparse(purl)
                sub_name = parsed.path.strip("/").replace("/", "_")
                fname = sub_name if sub_name.endswith(".pdf") else f"{os.path.basename(parsed.path)}"
                dest = rg_dir / fname
                if dest.exists() and dest.stat().st_size > 1024:
                    print(f"[OK] lecture_notes/roughgarden/{fname} ({format_size(dest.stat().st_size)}) [Önbellek]")
                    pdf_stats["total"] += 1
                    if extract_text_from_pdf(dest):
                        pdf_stats["success"] += 1
                else:
                    ok, _ = smart_download(session, purl, dest, timeout=40)
                    if ok:
                        print(f"[OK] lecture_notes/roughgarden/{fname} ({format_size(dest.stat().st_size)})")
                        pdf_stats["total"] += 1
                        if extract_text_from_pdf(dest):
                            pdf_stats["success"] += 1
                    else:
                        print(f"[SKIP] lecture_notes/roughgarden/{fname} (İndirilemedi)")
                        skip_list.append({"category": "lecture_notes", "url": purl, "reason": "Roughgarden fail"})
                check_pdf_quality(dest, low_quality_list)
    except Exception as e:
        print(f"[SKIP] lecture_notes/{rg_url} ({e})")
        skip_list.append({"category": "lecture_notes", "url": rg_url, "reason": str(e)})

    # 4. MIT 6.046 - Design and Analysis of Algorithms (Typed Lecture Notes)
    m46_dir = BASE_DIR / "lecture_notes" / "mit_6046"
    m46_dir.mkdir(parents=True, exist_ok=True)
    m46_url = "https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/pages/lecture-notes/"
    try:
        r = session.get(m46_url, timeout=20, headers=RETRY_HEADERS)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "lxml")
            res_pages = set()
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "resources/lecture-" in href and "handwritten" not in href:
                    res_pages.add(urljoin(m46_url, href))

            for page_u in sorted(res_pages):
                try:
                    p_resp = session.get(page_u, timeout=15, headers=RETRY_HEADERS)
                    if p_resp.status_code == 200:
                        p_soup = BeautifulSoup(p_resp.text, "lxml")
                        pdfs = [urljoin(page_u, a["href"]) for a in p_soup.find_all("a", href=True) if a["href"].endswith(".pdf")]
                        if pdfs:
                            pdf_u = pdfs[0]
                            fname = os.path.basename(urlparse(pdf_u).path)
                            dest = m46_dir / fname
                            if dest.exists() and dest.stat().st_size > 1024:
                                print(f"[OK] lecture_notes/mit_6046/{fname} ({format_size(dest.stat().st_size)}) [Önbellek]")
                                pdf_stats["total"] += 1
                                if extract_text_from_pdf(dest):
                                    pdf_stats["success"] += 1
                            else:
                                ok, _ = smart_download(session, pdf_u, dest, timeout=40)
                                if ok:
                                    print(f"[OK] lecture_notes/mit_6046/{fname} ({format_size(dest.stat().st_size)})")
                                    pdf_stats["total"] += 1
                                    if extract_text_from_pdf(dest):
                                        pdf_stats["success"] += 1
                                else:
                                    print(f"[SKIP] lecture_notes/mit_6046/{fname} (İndirilemedi)")
                                    skip_list.append({"category": "lecture_notes", "url": pdf_u, "reason": "6.046 fail"})
                            check_pdf_quality(dest, low_quality_list)
                except Exception:
                    continue
    except Exception as e:
        print(f"[SKIP] lecture_notes/{m46_url} ({e})")
        skip_list.append({"category": "lecture_notes", "url": m46_url, "reason": str(e)})

    # 5. Jeff Erickson - Algoritmalar Ders Bölümleri ve Notları
    je_dir = BASE_DIR / "lecture_notes" / "erickson_algorithms"
    je_dir.mkdir(parents=True, exist_ok=True)
    je_url = "https://jeffe.cs.illinois.edu/teaching/algorithms/"
    try:
        r = session.get(je_url, timeout=20, headers=RETRY_HEADERS)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "lxml")
            je_links = set()
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf") and ("book/" in href.lower() or "notes/" in href.lower()) and "!!" not in href:
                    je_links.add(urljoin(je_url, href))

            for purl in sorted(je_links):
                fname = os.path.basename(urlparse(purl).path)
                dest = je_dir / fname
                if dest.exists() and dest.stat().st_size > 1024:
                    print(f"[OK] lecture_notes/erickson_algorithms/{fname} ({format_size(dest.stat().st_size)}) [Önbellek]")
                    pdf_stats["total"] += 1
                    if extract_text_from_pdf(dest):
                        pdf_stats["success"] += 1
                else:
                    ok, _ = smart_download(session, purl, dest, timeout=40)
                    if ok:
                        print(f"[OK] lecture_notes/erickson_algorithms/{fname} ({format_size(dest.stat().st_size)})")
                        pdf_stats["total"] += 1
                        if extract_text_from_pdf(dest):
                            pdf_stats["success"] += 1
                    else:
                        print(f"[SKIP] lecture_notes/erickson_algorithms/{fname} (İndirilemedi)")
                        skip_list.append({"category": "lecture_notes", "url": purl, "reason": "Erickson fail"})
                check_pdf_quality(dest, low_quality_list)
    except Exception as e:
        print(f"[SKIP] lecture_notes/{je_url} ({e})")
        skip_list.append({"category": "lecture_notes", "url": je_url, "reason": str(e)})

    # 6. MIT 6.042 Ders Notu Kontrolü
    mit_6042_url = "https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/pages/lecture-notes/"
    try:
        r = session.get(mit_6042_url, timeout=15, headers=RETRY_HEADERS)
        if r.status_code != 200:
            print(f"[SKIP] lecture_notes/{mit_6042_url} (HTTP {r.status_code} redirect)")
            skip_list.append({"category": "lecture_notes", "url": mit_6042_url, "reason": f"HTTP {r.status_code}"})
    except Exception as e:
        print(f"[SKIP] lecture_notes/{mit_6042_url} ({e})")
        skip_list.append({"category": "lecture_notes", "url": mit_6042_url, "reason": str(e)})


def process_theory_expansion(session: requests.Session, skip_list: List[Dict], pdf_stats: Dict, low_quality_list: List[Dict]):
    """SORUN 3 (Theory): MIT 6.045 ders notları ve Princeton/Brown denetimleri."""
    logger.info(">>> Hesaplama Teorisi Genişletmesi (MIT 6.045, Princeton, Brown) İşleniyor...")
    th_dir = BASE_DIR / "theory" / "mit_6045"
    th_dir.mkdir(parents=True, exist_ok=True)

    # 1. MIT 6.045 - Automata, Computability, and Complexity
    mit_6045_url = "https://ocw.mit.edu/courses/6-045j-automata-computability-and-complexity-spring-2011/pages/lecture-notes/"
    try:
        r = session.get(mit_6045_url, timeout=20, headers=RETRY_HEADERS)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "lxml")
            resource_urls = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "resources/mit6_045js11_lec" in href:
                    resource_urls.append(urljoin(mit_6045_url, href))

            for res_url in sorted(set(resource_urls)):
                try:
                    res_resp = session.get(res_url, timeout=15, headers=RETRY_HEADERS)
                    if res_resp.status_code == 200:
                        res_soup = BeautifulSoup(res_resp.text, "lxml")
                        pdf_a = [urljoin(res_url, a["href"]) for a in res_soup.find_all("a", href=True) if a["href"].endswith(".pdf")]
                        if pdf_a:
                            pdf_url = pdf_a[0]
                            fname = os.path.basename(urlparse(pdf_url).path)
                            dest = th_dir / fname
                            if dest.exists() and dest.stat().st_size > 1024:
                                print(f"[OK] theory/mit_6045/{fname} ({format_size(dest.stat().st_size)}) [Önbellek]")
                                pdf_stats["total"] += 1
                                if extract_text_from_pdf(dest):
                                    pdf_stats["success"] += 1
                            else:
                                ok, _ = smart_download(session, pdf_url, dest, timeout=40)
                                if ok:
                                    print(f"[OK] theory/mit_6045/{fname} ({format_size(dest.stat().st_size)})")
                                    pdf_stats["total"] += 1
                                    if extract_text_from_pdf(dest):
                                        pdf_stats["success"] += 1
                                else:
                                    print(f"[SKIP] theory/mit_6045/{fname} (İndirilemedi)")
                                    skip_list.append({"category": "theory", "url": pdf_url, "reason": "6.045 PDF fail"})
                            check_pdf_quality(dest, low_quality_list)
                except Exception:
                    continue
        else:
            print(f"[SKIP] theory/{mit_6045_url} (HTTP {r.status_code})")
            skip_list.append({"category": "theory", "url": mit_6045_url, "reason": f"HTTP {r.status_code}"})
    except Exception as e:
        print(f"[SKIP] theory/{mit_6045_url} ({e})")
        skip_list.append({"category": "theory", "url": mit_6045_url, "reason": str(e)})

    # 2. Princeton COS 487 (404 kontrolü ve SKIP loglama)
    cos487_url = "https://www.cs.princeton.edu/courses/archive/fall04/cos487/lectures/"
    try:
        r = session.get(cos487_url, timeout=10, headers=RETRY_HEADERS)
        if r.status_code == 404:
            print(f"[SKIP] theory/{cos487_url} (HTTP 404 Not Found)")
            skip_list.append({"category": "theory", "url": cos487_url, "reason": "HTTP 404 Not Found"})
        elif r.status_code == 200:
            print(f"[OK] theory/{cos487_url} (200)")
    except Exception as e:
        print(f"[SKIP] theory/{cos487_url} ({e})")
        skip_list.append({"category": "theory", "url": cos487_url, "reason": str(e)})

    # 3. Brown CSCI 1010
    brown_url = "https://cs.brown.edu/courses/csci1010/"
    try:
        r = session.get(brown_url, timeout=10, headers=RETRY_HEADERS)
        if r.status_code == 200:
            print(f"[SKIP] theory/{brown_url} (Doğrudan PDF ders notu yok, telifli Sipser yönlendirmesi)")
            skip_list.append({"category": "theory", "url": brown_url, "reason": "No direct open PDF notes"})
    except Exception as e:
        print(f"[SKIP] theory/{brown_url} ({e})")
        skip_list.append({"category": "theory", "url": brown_url, "reason": str(e)})


def process_networking_expansion(session: requests.Session, skip_list: List[Dict]):
    """SORUN 3 (Networking): Kurose & Ross ders notları ve slayt kontrolleri."""
    logger.info(">>> Bilgisayar Ağları Genişletmesi (Kurose & Ross) İşleniyor...")
    net_dir = BASE_DIR / "networking"
    net_dir.mkdir(parents=True, exist_ok=True)

    # 1. Kurose & Ross Online Lectures HTML
    kr_url = "https://gaia.cs.umass.edu/kurose_ross/online_lectures.htm"
    dest_kr = net_dir / "kurose_ross_online_lectures.html"
    try:
        r = session.get(kr_url, timeout=15, headers=RETRY_HEADERS)
        if r.status_code == 200:
            with open(dest_kr, "wb") as f:
                f.write(r.content)
            extract_text_from_html(dest_kr)
            print(f"[OK] networking/kurose_ross_online_lectures.html ({format_size(len(r.content))})")
        else:
            print(f"[SKIP] networking/{kr_url} (HTTP {r.status_code})")
            skip_list.append({"category": "networking", "url": kr_url, "reason": f"HTTP {r.status_code}"})
    except Exception as e:
        print(f"[SKIP] networking/{kr_url} ({e})")
        skip_list.append({"category": "networking", "url": kr_url, "reason": str(e)})

    # 2. Kurose & Ross PPT Slaytları (404 kontrolü)
    ppt_url = "https://www-net.cs.umass.edu/kurose-ross-ppt-8e/"
    try:
        r = session.get(ppt_url, timeout=10, headers=RETRY_HEADERS)
        if r.status_code != 200:
            print(f"[SKIP] networking/{ppt_url} (HTTP {r.status_code} Not Found)")
            skip_list.append({"category": "networking", "url": ppt_url, "reason": f"HTTP {r.status_code}"})
    except Exception as e:
        print(f"[SKIP] networking/{ppt_url} ({e})")
        skip_list.append({"category": "networking", "url": ppt_url, "reason": str(e)})


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


def process_papers_we_love(skip_list: List[Dict], pdf_stats: Dict, low_quality_list: List[Dict]):
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
            check_pdf_quality(pdf_path, low_quality_list)
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
                            check_pdf_quality(dest_file, low_quality_list)

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


def process_turkish_corpus(session: requests.Session, pdf_stats: Dict, low_quality_list: List[Dict]):
    """Türkçe CS Wikipedia sayfalarını, açık erişimli tezleri ve ders notlarını toplar."""
    logger.info(">>> Türkçe CS Veri Seti (tr_docs) Toplanıyor...")
    tr_dir = BASE_DIR / "tr_docs"
    tr_dir.mkdir(parents=True, exist_ok=True)
    write_license_note(tr_dir, "tr_docs")

    # 1. Türkçe Wikipedia CS Sayfaları
    wiki_keywords = [
        "bilgisayar bilimi", "algoritma", "veri yapısı", "programlama dili", "işletim sistemi",
        "bilgisayar mimarisi", "bilgisayar ağları", "veritabanı yönetim sistemi", "yapay zeka",
        "makine öğrenimi", "derleyici", "nesne yönelimli programlama", "ayrık matematik",
        "yığın veri yapısı", "kuyruk veri yapısı", "bağlı liste", "ikili arama ağacı",
        "sıralama algoritması", "kriptografi", "bellek yönetimi", "süreç bilişim", "iş parçacığı",
        "dağıtık sistemler", "yazılım mühendisliği", "graf teorisi", "özyineleme", "arama algoritması",
        "turing makinesi", "biçimsel diller", "bilişim", "mikroişlemci", "önbellek", "sanal bellek",
        "dinamik programlama", "açgözlü algoritma", "dijkstra algoritması", "hızlı sıralama"
    ]
    wiki_url = "https://tr.wikipedia.org/w/api.php"
    wiki_headers = {"User-Agent": "MergenSNNCorpus/1.0 (https://github.com/haremir/mergen-snn; contact: info@mergen.ai)"}
    titles: Set[str] = set()

    for kw in wiki_keywords:
        try:
            params = {"action": "query", "list": "search", "srsearch": kw, "format": "json", "srlimit": 15}
            r = requests.get(wiki_url, params=params, headers=wiki_headers, timeout=10).json()
            for item in r.get("query", {}).get("search", []):
                t = item.get("title")
                if t and not any(t.startswith(p) for p in ["Kategori:", "Dosya:", "Şablon:", "Tartışma:"]):
                    titles.add(t)
        except Exception as e:
            logger.warning(f"Wikipedia arama hatası ({kw}): {e}")

    logger.info(f"Türkçe Wikipedia'dan {len(titles)} benzersiz CS başlığı bulundu.")
    wiki_saved = 0
    for title in sorted(titles):
        safe_name = re.sub(r'[\/:*?"<>|\s]+', '_', title).strip('_')
        out_file = tr_dir / f"wiki_{safe_name}.txt"
        if out_file.exists() and out_file.stat().st_size > 200:
            wiki_saved += 1
            continue

        try:
            time.sleep(0.05)
            params = {"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "format": "json"}
            res = requests.get(wiki_url, params=params, headers=wiki_headers, timeout=10).json()
            pages = res.get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                content = pdata.get("extract", "")
                if content and len(content.strip()) > 200:
                    out_file.write_text(content.strip(), encoding="utf-8")
                    wiki_saved += 1
                    print(f"[OK] tr_docs/wiki_{safe_name}.txt ({format_size(len(content))})")
        except Exception as e:
            logger.warning(f"Wikipedia metin çekme hatası ({title}): {e}")

    logger.info(f"Türkçe Wikipedia toplam makale: {wiki_saved}")

    # 2. Açık Erişim İTÜ Polen / DSpace Tezleri ve Ders Notları
    theses_dir = tr_dir / "theses"
    theses_dir.mkdir(parents=True, exist_ok=True)
    dspace_queries = [
        "bilgisayar mühendisliği",
        "algoritma veri yapıları",
        "yapay zeka derin öğrenme",
        "yazılım mühendisliği",
    ]
    theses_saved = 0
    max_theses = 35

    for query in dspace_queries:
        if theses_saved >= max_theses:
            break
        try:
            ds_url = f"https://polen.itu.edu.tr/server/api/discover/search/objects?query={requests.utils.quote(query)}&size=20"
            r = session.get(ds_url, verify=False, timeout=15).json()
            objects = r.get("_embedded", {}).get("searchResult", {}).get("_embedded", {}).get("objects", [])

            for obj in objects:
                if theses_saved >= max_theses:
                    break
                res = obj.get("_embedded", {}).get("indexableObject", {})
                bundles_url = res.get("_links", {}).get("bundles", {}).get("href")
                if not bundles_url:
                    continue

                r_b = session.get(bundles_url, verify=False, timeout=10).json()
                for b in r_b.get("_embedded", {}).get("bundles", []):
                    if b.get("name") == "ORIGINAL":
                        r_bs = session.get(b["_links"]["bitstreams"]["href"], verify=False, timeout=10).json()
                        for bs in r_bs.get("_embedded", {}).get("bitstreams", []):
                            bs_name = bs.get("name", "")
                            if bs_name.lower().endswith(".pdf"):
                                item_id = bs.get("id", str(theses_saved))
                                dest_pdf = theses_dir / f"thesis_{item_id[:8]}.pdf"
                                dest_txt = dest_pdf.with_suffix(".txt")

                                if dest_txt.exists() and dest_txt.stat().st_size > 1024:
                                    theses_saved += 1
                                    continue

                                content_url = bs["_links"]["content"]["href"]
                                ok, _ = smart_download(session, content_url, dest_pdf, timeout=60)
                                if ok:
                                    pdf_stats["total"] += 1
                                    if extract_text_from_pdf(dest_pdf):
                                        pdf_stats["success"] += 1
                                        check_pdf_quality(dest_pdf, low_quality_list)
                                        theses_saved += 1
                                        print(f"[OK] tr_docs/theses/{dest_txt.name} ({format_size(dest_txt.stat().st_size)})")
                                break
        except Exception as e:
            logger.warning(f"DSpace tez indirme hatası ({query}): {e}")

    logger.info(f"Türkçe açık erişim tez/doküman sayısı: {theses_saved}")


def process_code_corpus(session: requests.Session, target_count: int = 5000):
    """Python ve C dillerinde 5.000 dosyalık lisanslı kod korpusu indirir ve .txt olarak kaydeder."""
    logger.info(">>> Kod Ağırlıklı Veri Seti (code_docs) Toplanıyor...")
    code_dir = BASE_DIR / "code_docs"
    py_dir = code_dir / "python"
    c_dir = code_dir / "c"
    py_dir.mkdir(parents=True, exist_ok=True)
    c_dir.mkdir(parents=True, exist_ok=True)
    write_license_note(code_dir, "code_docs")

    existing_py = list(py_dir.glob("*.txt"))
    existing_c = list(c_dir.glob("*.txt"))
    if len(existing_py) + len(existing_c) >= target_count:
        logger.info(f"Kod korpusu önbellekte mevcut ({len(existing_py)} Python, {len(existing_c)} C). Atlanıyor.")
        return

    # 1. TheAlgorithms/Python Klonla
    temp_py = tempfile.mkdtemp(prefix="algo_py_")
    try:
        logger.info("TheAlgorithms/Python klonlanıyor...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/TheAlgorithms/Python.git", temp_py],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        py_files = sorted(Path(temp_py).rglob("*.py"))
        py_count = len(existing_py)
        for pf in py_files:
            if pf.name.startswith("__") or "test" in pf.name.lower():
                continue
            try:
                code_text = pf.read_text(encoding="utf-8", errors="ignore")
                if len(code_text.strip()) > 50:
                    out_path = py_dir / f"algo_py_{py_count:04d}_{pf.name}.txt"
                    if not out_path.exists():
                        out_path.write_text(code_text, encoding="utf-8")
                        py_count += 1
            except Exception:
                continue
        logger.info(f"TheAlgorithms/Python tamamlandı: {py_count} dosya.")
    except Exception as e:
        logger.warning(f"TheAlgorithms/Python hatası: {e}")
    finally:
        shutil.rmtree(temp_py, ignore_errors=True)

    # 2. TheAlgorithms/C Klonla
    temp_c = tempfile.mkdtemp(prefix="algo_c_")
    try:
        logger.info("TheAlgorithms/C klonlanıyor...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/TheAlgorithms/C.git", temp_c],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        c_files = sorted(Path(temp_c).rglob("*.[ch]"))
        c_count = len(existing_c)
        for cf in c_files:
            try:
                code_text = cf.read_text(encoding="utf-8", errors="ignore")
                if len(code_text.strip()) > 50:
                    out_path = c_dir / f"algo_c_{c_count:04d}_{cf.name}.txt"
                    if not out_path.exists():
                        out_path.write_text(code_text, encoding="utf-8")
                        c_count += 1
            except Exception:
                continue
        logger.info(f"TheAlgorithms/C tamamlandı: {c_count} dosya.")
    except Exception as e:
        logger.warning(f"TheAlgorithms/C hatası: {e}")
    finally:
        shutil.rmtree(temp_c, ignore_errors=True)

    # 3. Python Kod Örneklemi (HuggingFace flytech/python-codes-25k) -> 3000'e tamamla
    current_py = len(list(py_dir.glob("*.txt")))
    needed_py = max(0, 3000 - current_py)
    if needed_py > 0:
        logger.info(f"HuggingFace üzerinden {needed_py} adet Python kodu indiriliyor...")
        try:
            import json
            hf_url = "https://huggingface.co/datasets/flytech/python-codes-25k/resolve/main/python-codes-25k.jsonl"
            r = session.get(hf_url, stream=True, timeout=30)
            added_py = 0
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    code = data.get("output", "")
                    if code and len(code.strip()) > 80:
                        out_path = py_dir / f"hf_py_{added_py:04d}.txt"
                        if not out_path.exists():
                            out_path.write_text(code.strip(), encoding="utf-8")
                            added_py += 1
                            if added_py >= needed_py:
                                break
                except Exception:
                    continue
            logger.info(f"HuggingFace'den {added_py} Python dosyası eklendi.")
        except Exception as e:
            logger.warning(f"HF Python veri hatası: {e}")

    # 4. C Kod Örneklemi (xv6-riscv / git C algorithms) -> 2000'e tamamla
    current_c = len(list(c_dir.glob("*.txt")))
    needed_c = max(0, 2000 - current_c)
    if needed_c > 0:
        logger.info(f"Ek C algoritmaları indiriliyor ({needed_c} dosya)...")
        temp_c_extra = tempfile.mkdtemp(prefix="xv6_clone_")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/mit-pdos/xv6-riscv.git", temp_c_extra],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
            extra_files = sorted(Path(temp_c_extra).rglob("*.[ch]"))
            added_c = 0
            for ef in extra_files:
                try:
                    code_text = ef.read_text(encoding="utf-8", errors="ignore")
                    if len(code_text.strip()) > 50:
                        out_path = c_dir / f"xv6_{added_c:04d}_{ef.name}.txt"
                        if not out_path.exists():
                            out_path.write_text(code_text, encoding="utf-8")
                            added_c += 1
                            if added_c >= needed_c:
                                break
                except Exception:
                    continue
            logger.info(f"xv6-riscv'den {added_c} C dosyası eklendi.")
        except Exception as e:
            logger.warning(f"xv6 klonlama hatası: {e}")
        finally:
            shutil.rmtree(temp_c_extra, ignore_errors=True)

    current_c = len(list(c_dir.glob("*.txt")))
    needed_c = max(0, 2000 - current_c)
    if needed_c > 0:
        temp_git = tempfile.mkdtemp(prefix="git_clone_")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/git/git.git", temp_git],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
            git_c_files = sorted(Path(temp_git).rglob("*.[ch]"))
            added_c = 0
            for gf in git_c_files:
                try:
                    code_text = gf.read_text(encoding="utf-8", errors="ignore")
                    if len(code_text.strip()) > 50:
                        out_path = c_dir / f"git_{added_c:04d}_{gf.name}.txt"
                        if not out_path.exists():
                            out_path.write_text(code_text, encoding="utf-8")
                            added_c += 1
                            if added_c >= needed_c:
                                break
                except Exception:
                    continue
            logger.info(f"git/git'den {added_c} C dosyası eklendi.")
        except Exception as e:
            logger.warning(f"git repo klonlama hatası: {e}")
        finally:
            shutil.rmtree(temp_git, ignore_errors=True)

    current_c = len(list(c_dir.glob("*.txt")))
    needed_c = max(0, 2000 - current_c)
    if needed_c > 0:
        temp_curl = tempfile.mkdtemp(prefix="curl_clone_")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/curl/curl.git", temp_curl],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
            curl_c_files = sorted(Path(temp_curl).rglob("*.[ch]"))
            added_c = 0
            for cf in curl_c_files:
                try:
                    code_text = cf.read_text(encoding="utf-8", errors="ignore")
                    if len(code_text.strip()) > 50:
                        out_path = c_dir / f"curl_{added_c:04d}_{cf.name}.txt"
                        if not out_path.exists():
                            out_path.write_text(code_text, encoding="utf-8")
                            added_c += 1
                            if added_c >= needed_c:
                                break
                except Exception:
                    continue
            logger.info(f"curl/curl'den {added_c} C dosyası eklendi.")
        except Exception as e:
            logger.warning(f"curl repo klonlama hatası: {e}")
        finally:
            shutil.rmtree(temp_curl, ignore_errors=True)

    total_code = len(list(py_dir.glob("*.txt"))) + len(list(c_dir.glob("*.txt")))
    logger.info(f"Kod korpusu tamamlandı: Toplam {total_code} dosya ({len(list(py_dir.glob('*.txt')))} Python, {len(list(c_dir.glob('*.txt')))} C).")


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
    low_quality_list: List[Dict] = []
    pdf_stats = {"total": 0, "success": 0}

    # 1. Doğrudan Dosyalar (Patterson & Hennessy, C++, Postgres, D2L, Sutton & Barto, RFC'ler vb.)
    process_direct_files(session, skip_list, pdf_stats, low_quality_list)

    # 2. Hailperin OS Kitabı (GitHub Clone -> systems/)
    process_hailperin_os(pdf_stats, low_quality_list)

    # 3. MIT 6.006 Algoritmalar Ders Notları (lec01 - lec24 -> lecture_notes/)
    process_mit_6006_lectures(session, pdf_stats, low_quality_list)

    # 4. Açık Erişim Mimari Kitapları (IIT Delhi & Rowan)
    process_architecture_open_access(session, skip_list, pdf_stats, low_quality_list)

    # 5. SORUN 1: gc_papers (PWL garbage_collection klasörünü aktar)
    process_gc_papers(pdf_stats, low_quality_list)

    # 6. SORUN 2: lecture_notes genişletmesi (Kleinberg-Tardos, Stanford CS161, Roughgarden)
    process_lecture_notes_expansion(session, skip_list, pdf_stats, low_quality_list)

    # 7. SORUN 3: theory genişletmesi (MIT 6.045, Princeton, Brown)
    process_theory_expansion(session, skip_list, pdf_stats, low_quality_list)

    # 8. SORUN 3: networking genişletmesi (Kurose & Ross)
    process_networking_expansion(session, skip_list)

    # 9. HTML tabanlı kitaplar (Crafting Interpreters, BottomUpCS, RedBook, Python Docs vb.)
    process_html_crawler(session, skip_list)

    # 10. Papers We Love makale arşivi
    process_papers_we_love(skip_list, pdf_stats, low_quality_list)

    # 11. Yerel telifli kitap denetimi
    check_local_required()

    # 12. Türkçe CS korpusu (tr_docs: Vikipedi ve açık erişim üniversite tezleri)
    process_turkish_corpus(session, pdf_stats, low_quality_list)

    # 13. Kod ağırlıklı veri seti (code_docs: 5.000 Python ve C dosyası)
    process_code_corpus(session, target_count=5000)

    # 14. Düşük kaliteli PDF raporunu kaydet
    save_low_quality_report(low_quality_list)

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
        txt_files = [f for f in all_files if f.suffix.lower() == ".txt" and f.name != "LICENSE_NOTE.txt" and f.name != "_low_quality.txt"]

        cat_total_size = sum(f.stat().st_size for f in all_files if f.name != "_low_quality.txt")
        cat_txt_size = sum(f.stat().st_size for f in txt_files)

        total_all_bytes += cat_total_size
        total_txt_bytes += cat_txt_size
        total_file_count += len([f for f in all_files if f.name != "_low_quality.txt"])
        total_txt_count += len(txt_files)

        print(
            f"- {cat:<20}: {len([f for f in all_files if f.name != '_low_quality.txt']):>3} dosya ({format_size(cat_total_size):>9}) | "
            f".txt: {len(txt_files):>3} adet ({format_size(cat_txt_size):>9})"
        )

    print("-" * 70)
    print(f"Toplam Dosya Sayısı       : {total_file_count}")
    print(f"Toplam Çıkarılan .txt     : {total_txt_count}")
    print(f"Toplam Ham + TXT Boyutu   : {format_size(total_all_bytes)}")
    print(f"TOPLAM ÇIKARILAN METİN    : {format_size(total_txt_bytes)} ({total_txt_bytes / (1024 * 1024):.2f} MB)")
    hedef_durumu = "[BASARILI - HEDEF ASILDI]" if total_txt_bytes >= 85 * 1024 * 1024 else "[85 MB ALTINDA]"
    print(f"Hedef Kontrolü (>= 85 MB) : {hedef_durumu}")
    print(f"Kalan SKIP / Hata Sayısı  : {len(skip_list)}")
    if pdf_stats["total"] > 0:
        pct = (pdf_stats["success"] / pdf_stats["total"]) * 100
        print(f"PDF Extraction Başarısı   : {pdf_stats['success']}/{pdf_stats['total']} (%{pct:.1f})")
    print(f"Düşük Kaliteli PDF Sayısı : {len(low_quality_list)}")
    print(f"Toplam Çalışma Süresi     : {elapsed:.1f} saniye")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

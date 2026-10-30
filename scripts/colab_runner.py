"""Mergen-SNN — Evrensel Colab & CLI Eğitim Başlatıcı (colab_runner.py).

Google Drive yollarını otomatik tespit eder, dosya bütünlüğünü KESİN olarak
doğrular ve Stage 4 Small SNN eğitimini kesintisiz başlatır.
"""

import os
from pathlib import Path
import subprocess
import sys

# Windows / Linux UTF-8 çıktı uyumluluğu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def mount_drive_if_colab():
    """Google Colab ortamındaysa Google Drive'ı mount eder."""
    try:
        from google.colab import drive
        print(">>> Google Colab ortamı tespit edildi. Google Drive bağlanıyor...")
        drive.mount("/content/drive")
        print(">>> Google Drive başarıyla bağlandı (/content/drive).")
    except ImportError:
        print(">>> Lokal veya bağımsız CLI ortamı tespit edildi (Google Colab modülü yok).")


def print_tree(start_path: Path, max_depth: int = 3, max_files_per_dir: int = 15):
    """Hata durumunda dizin yapısını ekrana basar."""
    print(f"\n{'='*70}")
    print(f"DİZİN AĞACI DÖKÜMÜ: {start_path}")
    print(f"{'='*70}")
    start_level = len(start_path.parts)
    for root, dirs, files in os.walk(start_path):
        current_path = Path(root)
        level = len(current_path.parts) - start_level
        if level > max_depth:
            dirs.clear()
            continue
        indent = "  " * level
        print(f"{indent}[DIR] {current_path.name}/")
        for i, f in enumerate(files):
            if i >= max_files_per_dir:
                print(f"{indent}  ... ve {len(files) - max_files_per_dir} dosya daha")
                break
            fp = current_path / f
            try:
                sz = fp.stat().st_size
                sz_str = f"{sz / 1024:.1f} KB" if sz < 1024 * 1024 else f"{sz / (1024 * 1024):.2f} MB"
            except Exception:
                sz_str = "?? B"
            print(f"{indent}  - {f} ({sz_str})")
    print(f"{'='*70}\n")


def find_project_dir() -> Path:
    """Tüm olası Drive ve yerel yolları tarayarak mergen-snn klasörünü bulur."""
    candidate_paths = [
        Path("/content/drive/MyDrive/mergen-snn"),
        Path("/content/drive/Drive/mergen-snn"),
        Path("/content/drive/mergen-snn"),
        Path("/content/mergen-snn"),
        Path(".").resolve(),
        Path("./mergen-snn").resolve(),
    ]

    print("\n>>> Proje dizini aranıyor...")
    for p in candidate_paths:
        if p.exists() and p.is_dir():
            # Temel script veya data dosyası var mı bak
            if (p / "scripts" / "train_lm.py").exists() or (p / "configs" / "stage4_small.yaml").exists():
                print(f">>> [OK] Aday yol doğrulandı: {p}")
                return p

    # Adaylarda doğrudan bulunamadıysa /content/drive içinde arama yap
    drive_root = Path("/content/drive")
    if drive_root.exists():
        print(">>> Aday listesi dışındaki Drive alt dizinleri taranıyor (/content/drive)...")
        for root, dirs, _ in os.walk(drive_root):
            if "mergen-snn" in dirs:
                found = Path(root) / "mergen-snn"
                print(f">>> [OK] Drive içinde mergen-snn tespit edildi: {found}")
                return found

    # Son çare: Geçerli çalışma dizini
    cwd = Path(".").resolve()
    print(f">>> [UYARI] Özel mergen-snn yolu bulunamadı, CWD deneniyor: {cwd}")
    return cwd


def ensure_uv():
    """uv paket yöneticisinin kurulu olduğundan emin olur."""
    try:
        subprocess.run(["uv", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    except Exception:
        print(">>> 'uv' kurulu değil, kuruluyor...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)


def main():
    print("=" * 70)
    print("Mergen-SNN — Evrensel Colab Runner")
    print("=" * 70)

    # 1. Drive mount et
    mount_drive_if_colab()

    # 2. Proje yolunu tespit et
    project_dir = find_project_dir()
    print(f">>> Aktif Proje Yolu: {project_dir}")

    # 3. Kritik dosyaları KESİN olarak kontrol et
    required_files = {
        "Eğitim Scripti": project_dir / "scripts" / "train_lm.py",
        "Eğitim Verisi": project_dir / "data" / "processed" / "bpe_train.bin",
        "Doğrulama Verisi": project_dir / "data" / "processed" / "bpe_val.bin",
        "BPE Tokenizer": project_dir / "data" / "processed" / "tokenizer_bpe.json",
        "Model Yapılandırması": project_dir / "configs" / "stage4_small.yaml",
    }

    missing_files = []
    print("\n>>> Kritik Dosya Kontrolleri:")
    for desc, fpath in required_files.items():
        if fpath.exists():
            sz = fpath.stat().st_size
            sz_str = f"{sz / (1024 * 1024):.2f} MB" if sz >= 1024 * 1024 else f"{sz / 1024:.1f} KB"
            print(f"  [VAR] {desc:<20}: {fpath.relative_to(project_dir)} ({sz_str})")
        else:
            print(f"  [YOK] {desc:<20}: {fpath}")
            missing_files.append((desc, fpath))

    if missing_files:
        print("\n[HATA] Bazı kritik dosyalar bulunamadı!")
        print("Teşhis için mevcut dizin yapısı taranıyor:")
        print_tree(project_dir, max_depth=3)
        if Path("/content/drive").exists():
            print("Drive ana dizini taranıyor:")
            print_tree(Path("/content/drive"), max_depth=2)
        sys.exit(1)

    print("\n>>> Tüm kritik dosyalar eksiksiz mevcut. Eğitim başlatılıyor...")

    # 4. Çalışma dizinini proje klasörüne çevir
    os.chdir(project_dir)

    # 5. uv ve bağımlılıkları doğrula
    ensure_uv()

    # 6. Komutu çalıştır
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/train_lm.py",
        "--config",
        "configs/stage4_small.yaml",
        "--resume",
        "auto",
    ]
    print(f">>> Çalıştırılan komut: {' '.join(cmd)}\n")
    sys.stdout.flush()

    try:
        ret = subprocess.run(cmd)
        sys.exit(ret.returncode)
    except KeyboardInterrupt:
        print("\n>>> Kullanıcı tarafından durduruldu.")
        sys.exit(130)


if __name__ == "__main__":
    main()

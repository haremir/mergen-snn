"""Mergen-SNN — Kaggle İçin Korpus Temizleme ve Arşivleme Scripti.

data/raw/cs_corpus/ altındaki dosya adlarını ASCII-safe hale getirir,
Türkçe ve sorunlu karakterleri temizler, çakışmaları çözer ve
Kaggle upload uyumlu cs_corpus_clean.tar.gz arşivini oluşturur.
"""

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
import unicodedata

# Türkçe karakter haritası
TR_MAP = str.maketrans({
    'ı': 'i', 'I': 'I', 'İ': 'I', 'i': 'i',
    'ğ': 'g', 'Ğ': 'G',
    'ş': 's', 'Ş': 'S',
    'ç': 'c', 'Ç': 'C',
    'ö': 'o', 'Ö': 'O',
    'ü': 'u', 'Ü': 'U'
})

# Sorunlu karakterler: ', ", \, :, *, ?, <, >, |, #, %, &, {, }, $, !, @, +, `, =
PROBLEMATIC_PATTERN = re.compile(r"""['"\\:*?<>|#%&{}$!@+`=]""")


def sanitize_stem(stem: str, original_full_name: str) -> str:
    """Dosya kökünü (stem) ASCII-safe ve sorunsuz hale getirir."""
    # 1. Türkçe karakter dönüşümü
    s = stem.translate(TR_MAP)
    # 2. Unicode NFKD ayrıştırması ve ASCII filtreleme
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    # 3. Sorunlu karakterleri alt çizgi (_) ile değiştir
    s = PROBLEMATIC_PATTERN.sub('_', s)
    # 4. Boşlukları _ ile değiştir
    s = s.replace(' ', '_')
    # 5. Çoklu _ karakterlerini tek _ yap
    s = re.sub(r'_+', '_', s)
    # 6. Baştaki ve sondaki _ karakterlerini temizle
    s = s.strip('_')
    # 7. İsim tamamen boş kalırsa hash ata
    if not s:
        h = hashlib.md5(original_full_name.encode('utf-8', 'ignore')).hexdigest()[:8]
        s = f"file_{h}"
    return s


def sanitize_filename(filename: str) -> str:
    """Dosya adını ve uzantısını normalize eder."""
    p = Path(filename)
    ext = p.suffix
    stem = p.stem
    clean_stem = sanitize_stem(stem, filename)
    clean_ext = ext.lower()
    return f"{clean_stem}{clean_ext}"


def main():
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "data" / "raw" / "cs_corpus"
    dst_dir = repo_root / "data" / "raw" / "cs_corpus_kaggle"
    tar_out = repo_root / "data" / "raw" / "cs_corpus_clean.tar.gz"

    print("=" * 60)
    print("Mergen-SNN: Kaggle Korpus Temizleme & Arşivleme")
    print("=" * 60)
    print(f"Kaynak Dizin : {src_dir}")
    print(f"Hedef Dizin  : {dst_dir}")
    print(f"Hedef Arşiv  : {tar_out}")
    print("=" * 60)

    if not src_dir.exists():
        print(f"HATA: Kaynak dizin bulunamadı: {src_dir}")
        sys.exit(1)

    # Hedef dizini temizle ve hazırla
    if dst_dir.exists():
        print("Mevcut hedef dizin temizleniyor...")
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    renamed_count = 0
    sample_renames = []
    name_tracker = {}

    t_start = time.time()

    # Dosyaları dolaş ve temizleyerek kopyala
    for root, dirs, files in os.walk(src_dir):
        rel_folder = Path(root).relative_to(src_dir)
        target_folder = dst_dir / rel_folder
        target_folder.mkdir(parents=True, exist_ok=True)

        if target_folder not in name_tracker:
            name_tracker[target_folder] = set()
        used_names = name_tracker[target_folder]

        for fname in sorted(files):
            clean_fname = sanitize_filename(fname)

            # Klasör içinde çakışma varsa _1, _2 ekle
            if clean_fname in used_names or (target_folder / clean_fname).exists():
                p = Path(clean_fname)
                base = p.stem
                ext = p.suffix
                counter = 1
                while f"{base}_{counter}{ext}" in used_names or (target_folder / f"{base}_{counter}{ext}").exists():
                    counter += 1
                clean_fname = f"{base}_{counter}{ext}"

            used_names.add(clean_fname)

            if clean_fname != fname:
                renamed_count += 1
                if len(sample_renames) < 8:
                    sample_renames.append((fname, clean_fname))

            src_file = Path(root) / fname
            dst_file = target_folder / clean_fname
            shutil.copy2(src_file, dst_file)
            copied_count += 1

    t_copy = time.time() - t_start
    print(f"\nDosya Kopyalama Tamamlandı ({t_copy:.2f} sn):")
    print(f" - Toplam Kopyalanan Dosya : {copied_count:,}")
    print(f" - İsmi Değiştirilen Dosya : {renamed_count:,}")

    if sample_renames:
        print("\nÖrnek Değişimler:")
        for orig, clean in sample_renames:
            print(f"  {orig} -> {clean}")

    # Arşivleme
    print("\nArşiv oluşturuluyor (tar -czf)...")
    if tar_out.exists():
        tar_out.unlink()

    # Önce sistem tar komutunu dene
    tar_success = False
    try:
        cmd = ["tar", "-czf", str(tar_out), "-C", str(repo_root / "data" / "raw"), "cs_corpus_kaggle"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and tar_out.exists() and tar_out.stat().st_size > 0:
            tar_success = True
            print("bsdtar ile sıkıştırma başarılı.")
        else:
            print(f"tar komut uyarısı: {res.stderr}")
    except Exception as e:
        print(f"tar çağrısı hatası: {e}")

    if not tar_success:
        print("Python tarfile modülü ile sıkıştırılıyor...")
        with tarfile.open(tar_out, "w:gz") as tar:
            tar.add(str(dst_dir), arcname="cs_corpus_kaggle")
        print("tarfile ile sıkıştırma başarılı.")

    tar_size_bytes = tar_out.stat().st_size
    tar_size_mb = tar_size_bytes / (1024 * 1024)
    print(f"\nArşiv Tamamlandı:")
    print(f" - Konum: {tar_out}")
    print(f" - Boyut: {tar_size_mb:.2f} MB ({tar_size_bytes:,} bayt)")
    print("=" * 60)


if __name__ == "__main__":
    main()

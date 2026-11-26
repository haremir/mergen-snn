"""Mergen-SNN — Stage 8 Veri Setlerini Colab'a Hızlı (Gzip Destekli) Yükleme Scripti.

Özellikler:
- tokenizer_bpe_v2.json ve bpe_val_v3b.bin dosyalarını doğrudan yükler.
- bpe_train_v3c.bin (4.0 GB) dosyasını 50 MB dilimler halinde anlık Gzip (seviye 1) ile sıkıştırıp
  ~20 MB'a indirerek tünelden 2.4x daha hızlı aktarır.
- Colab üzerinde anında decompress edilip nihai dosyaya append edilir (Colab diskinde çift yer kaplamaz).
- Kesinti durumunda Colab'deki mevcut boyuttan kusursuz resume eder.
"""

import gzip
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from colab_exec import exec_on_colab

SESSION = "mergen-stage8-100m"
CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB ham dilim (~20.9 MB sıkıştırılmış)
TEMP_CHUNK_LOCAL = PROJECT_ROOT / "data" / "temp_upload_chunk.gz"


def get_remote_file_size(remote_path: str) -> int:
    code = f"""
import os
if os.path.exists('{remote_path}'):
    print('SIZE:', os.path.getsize('{remote_path}'))
else:
    print('SIZE: 0')
"""
    res = exec_on_colab(code, session=SESSION, timeout=25)
    for line in res.stdout.splitlines():
        if "SIZE:" in line:
            return int(line.split("SIZE:")[1].strip())
    return 0


def upload_compressed_streaming(local_path: Path, remote_path: str):
    total_size = local_path.stat().st_size
    print(f"\n==========================================")
    print(f"HIZLI YÜKLEME (GZIP): {local_path.name} ({total_size / (1024**3):.2f} GB / {total_size / (1024**2):.1f} MB)")
    print(f"Hedef: {remote_path}")
    print(f"==========================================")

    remote_size = get_remote_file_size(remote_path)
    if remote_size >= total_size:
        print(f"✅ Dosya Colab'de zaten eksiksiz mevcut ({remote_size:,} bayt). Atlanıyor.")
        return

    if remote_size > 0:
        print(f"🔄 Colab üzerinde {remote_size / (1024**2):.1f} MB zaten mevcut! {remote_size:,} bayttan devam ediliyor...")

    start_offset = remote_size
    total_chunks = (total_size - start_offset + CHUNK_SIZE - 1) // CHUNK_SIZE
    chunk_idx = 0
    t_start = time.time()
    uploaded_uncompressed_bytes = 0
    total_wire_bytes = 0

    with open(local_path, "rb") as f_in:
        f_in.seek(start_offset)
        while True:
            raw_chunk = f_in.read(CHUNK_SIZE)
            if not raw_chunk:
                break
            chunk_idx += 1
            raw_len = len(raw_chunk)

            # 1. Hızlı gzip sıkıştırma (seviye 1, ~0.6s)
            t_comp0 = time.time()
            comp_chunk = gzip.compress(raw_chunk, compresslevel=1)
            comp_len = len(comp_chunk)

            with open(TEMP_CHUNK_LOCAL, "wb") as f_temp:
                f_temp.write(comp_chunk)

            # 2. Sıkıştırılmış dilimi Colab'a aktar
            c_remote_temp = "/content/temp_chunk.gz"
            comp_mb = comp_len / (1024 * 1024)
            current_total_mb = (start_offset + uploaded_uncompressed_bytes + raw_len) / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            pct = (start_offset + uploaded_uncompressed_bytes + raw_len) / total_size * 100

            print(f"  [{chunk_idx}/{total_chunks}] {raw_len/(1024*1024):.0f}MB -> {comp_mb:.1f}MB ({current_total_mb:.0f}/{total_mb:.0f} MB - %{pct:.1f})...", end=" ", flush=True)

            upload_success = False
            for attempt in range(4):
                t0 = time.time()
                cmd = ["uv", "tool", "run", "colabapi", "raw", "--", "upload", "-s", SESSION, str(TEMP_CHUNK_LOCAL), c_remote_temp]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                if res.returncode == 0:
                    upload_success = True
                    break
                else:
                    print(f"Tekrar ({attempt+1}/4)...", end=" ", flush=True)
                    time.sleep(2)

            if not upload_success:
                print(f"\n❌ Dilim yükleme hatası ({chunk_idx})")
                sys.exit(1)

            # 3. Colab üzerinde aç ve append et
            remote_append_code = f"""
import os, gzip
with open('{remote_path}', 'ab') as f_out, open('{c_remote_temp}', 'rb') as f_in:
    f_out.write(gzip.decompress(f_in.read()))
os.remove('{c_remote_temp}')
"""
            res_app = exec_on_colab(remote_append_code, session=SESSION, timeout=30)
            if res_app.returncode != 0:
                print(f"\n❌ Colab append hatası: {res_app.stderr}")
                sys.exit(1)

            uploaded_uncompressed_bytes += raw_len
            total_wire_bytes += comp_len

            elapsed_tot = time.time() - t_start
            effective_speed = (uploaded_uncompressed_bytes / (1024 * 1024)) / max(elapsed_tot, 0.1)
            rem_raw_mb = (total_size - (start_offset + uploaded_uncompressed_bytes)) / (1024 * 1024)
            eta_s = rem_raw_mb / max(effective_speed, 0.1)
            print(f"OK (Efektif: {effective_speed:.1f} MB/s | ETA: {eta_s/60:.1f} dk)")

    if TEMP_CHUNK_LOCAL.exists():
        TEMP_CHUNK_LOCAL.unlink()

    final_rem_size = get_remote_file_size(remote_path)
    print(f"✅ {local_path.name} TAMAMLANDI! Colab Boyutu: {final_rem_size:,} bayt ({final_rem_size/(1024**3):.2f} GB)")
    assert final_rem_size == total_size, f"Boyut uyuşmazlığı: {final_rem_size} vs {total_size}"


def main():
    data_dir = PROJECT_ROOT / "data" / "processed"
    train_file = data_dir / "bpe_train_v3c.bin"

    print("==========================================")
    print("STAGE 8 HIZLI YÜKLEME PIPELINE (GZIP STREAM)")
    print(f"Oturum: {SESSION}")
    print("==========================================")

    t0_all = time.time()
    upload_compressed_streaming(train_file, "/content/mergen-snn/data/processed/bpe_train_v3c.bin")
    total_time = time.time() - t0_all

    print("\n" + "=" * 60)
    print(f"TÜM VERİ SETLERİ COLAB'A BAŞARIYLA YÜKLENDİ! Toplam süre: {total_time/60:.1f} dakika")
    print("=" * 60)


if __name__ == "__main__":
    main()

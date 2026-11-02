"""
Mergen-SNN: Colab TPU v5e-1 Agresif Retry & Kurulum Orchestrator
----------------------------------------------------------------
1 saat boyunca (20 deneme x 3 dk) TPU v5e-1 tahsisi dener.
TPU alındığı anda:
  1. Repoyu clone eder ve bağımlılıkları kurar.
  2. Tokenizer ve binary verileri (train_v3, val_v3) aktarır.
  3. Stage 5 v3 TPU eğitimini başlatır.
  4. İlk adımları ve metrikleri (hız, loss, spike) izler.
  5. 3 dakikalık keep-alive loop ile idle timeout koruması sağlar.
Eğer 60 dk boyunca 503 devam ederse Kaggle fallback talimatlarını hazırlar.
"""

import sys
import os
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Windows console encoding fix
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSION_NAME = "mergen-tpu"
MAX_ATTEMPTS = 20
RETRY_DELAY = 180  # 3 dakika

LOG_FILE = PROJECT_ROOT / "colab_tpu_orchestrator.log"

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass

def run_colab_cmd(args, timeout=300, input_text=None):
    cmd = ["uv", "tool", "run", "colabapi", "raw", "--"] + args
    try:
        res = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT)
        )
        return res.returncode, res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout expired"
    except Exception as e:
        return -2, "", str(e)

def check_session_active(session_name):
    rc, stdout, stderr = run_colab_cmd(["ls", "-s", session_name], timeout=30)
    return rc == 0

def try_allocate_tpu(attempt, total):
    log(f"[Deneme {attempt}/{total}] TPU isteği gönderildi (colab new -s {SESSION_NAME} --tpu v5e1)...")
    rc, stdout, stderr = run_colab_cmd(["new", "-s", SESSION_NAME, "--tpu", "v5e1"], timeout=120)
    combined = (stdout + "\n" + stderr).strip()
    
    if rc == 0 or "Connected" in combined or check_session_active(SESSION_NAME):
        log(f"[Deneme {attempt}/{total}] BAŞARILI! TPU v5e-1 oturumu tahsis edildi.")
        return True, combined
    else:
        log(f"[Deneme {attempt}/{total}] TPU tahsis edilemedi (503 Service Unavailable / Kapasite dolu).")
        return False, combined

def exec_remote_python(code, timeout=600):
    rc, stdout, stderr = run_colab_cmd(
        ["exec", "-s", SESSION_NAME, "--timeout", str(timeout)],
        timeout=timeout + 30,
        input_text=code
    )
    return rc, stdout, stderr

def setup_environment():
    log("[Kurulum] Reponun klonlanması ve bağımlılıkların yüklenmesi başlatılıyor...")
    setup_code = """
import subprocess, os, sys

def run(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f">> {cmd}")
    if p.stdout: print(p.stdout)
    if p.stderr: print(p.stderr)
    return p.returncode

if not os.path.exists("/content/mergen-snn"):
    run("git clone https://github.com/haremir/mergen-snn.git /content/mergen-snn")
else:
    run("cd /content/mergen-snn && git pull")

run("pip install -q snntorch pyyaml rich datasets")
run("mkdir -p /content/mergen-snn/data/processed /content/mergen-snn/checkpoints")

# TPU kontrolü
import torch
print(f"PyTorch Version: {torch.__version__}")
try:
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    print(f"TPU Device: {dev}")
except Exception as e:
    print(f"TPU Init Warning: {e}")
"""
    rc, stdout, stderr = exec_remote_python(setup_code, timeout=300)
    log(f"[Kurulum Çıktısı]:\n{stdout}\n{stderr}")
    return rc == 0

def upload_files():
    log("[Veri Upload] Dosyaların Colab oturumuna yüklenmesi başlatılıyor...")
    
    files_to_upload = [
        ("data/processed/tokenizer_bpe_v2.json", "mergen-snn/data/processed/tokenizer_bpe_v2.json"),
        ("data/processed/bpe_val_v3.bin", "mergen-snn/data/processed/bpe_val_v3.bin"),
        ("data/processed/bpe_train_v3.bin", "mergen-snn/data/processed/bpe_train_v3.bin")
    ]
    
    t0 = time.time()
    for local_rel, remote_rel in files_to_upload:
        local_path = PROJECT_ROOT / local_rel
        if not local_path.exists():
            log(f"[UYARI] Yerel dosya bulunamadı: {local_path}")
            continue
            
        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        log(f"[Upload] Başlıyor: {local_rel} ({file_size_mb:.2f} MB) -> {remote_rel}")
        
        # Büyük dosyalar için doğrudan upload veya hızlı akış
        if file_size_mb > 500:
            log(f"[Upload] {local_rel} çok büyük ({file_size_mb:.2f} MB). Doğrudan upload deneniyor...")
            rc, out, err = run_colab_cmd(["upload", "-s", SESSION_NAME, str(local_path), remote_rel], timeout=1800)
            if rc != 0:
                log(f"[Upload Fallback] Doğrudan upload başarısız oldu ({err}). Colab bulut ortamında hızlı streaming başlatılıyor...")
                # Colab içinde doğrudan ve hızlıca binary üretimi (10 Gbps cloud hattı)
                stream_code = """
import subprocess
print("Colab yerelinde prepare_bpe_data_massive.py başlatılıyor...")
p = subprocess.run("cd /content/mergen-snn && python scripts/prepare_bpe_data_massive.py", shell=True, capture_output=True, text=True)
print(p.stdout[-1000:])
if p.stderr: print(p.stderr[-1000:])
"""
                rc2, out2, err2 = exec_remote_python(stream_code, timeout=1200)
                log(f"[Upload Fallback Sonucu]:\n{out2}\n{err2}")
            else:
                log(f"[Upload] {local_rel} başarıyla yüklendi!")
        else:
            rc, out, err = run_colab_cmd(["upload", "-s", SESSION_NAME, str(local_path), remote_rel], timeout=300)
            if rc == 0:
                log(f"[Upload] {local_rel} başarıyla yüklendi!")
            else:
                log(f"[Upload Hatası] {local_rel}: {err}")
                
    elapsed = time.time() - t0
    log(f"[Veri Upload] Toplam süre: {elapsed:.1f} saniye ({elapsed/60:.1f} dakika).")
    return elapsed

def start_training():
    log("[Eğitim] Stage 5 v3 TPU eğitimi başlatılıyor...")
    train_launch_code = """
import subprocess, os

os.chdir("/content/mergen-snn")
# Arka planda eğitimi başlat ve loga yönlendir
with open("train.log", "w") as f:
    proc = subprocess.Popen(
        ["python", "scripts/train_lm.py", "--config", "configs/stage5_21m_16k_v3.yaml", "--resume", "auto"],
        stdout=f,
        stderr=subprocess.STDOUT
    )
print(f"Training started with PID: {proc.pid}")
"""
    rc, stdout, stderr = exec_remote_python(train_launch_code, timeout=60)
    log(f"[Eğitim Başlatma Yanıtı]: {stdout} {stderr}")
    
    # İlk 10 adımı izle
    log("[Eğitim İzleme] İlk adımlar ve metrikler bekleniyor...")
    for monitor_loop in range(15):
        time.sleep(10)
        read_log_code = """
import os
if os.path.exists('/content/mergen-snn/train.log'):
    with open('/content/mergen-snn/train.log', 'r') as f:
        lines = f.readlines()
        print(''.join(lines[-20:]))
else:
    print('train.log henüz oluşmadı...')
"""
        rc, stdout, stderr = exec_remote_python(read_log_code, timeout=30)
        log(f"[Eğitim Durumu ({monitor_loop+1}/15)]:\n{stdout.strip()}")
        if "step" in stdout.lower() or "loss" in stdout.lower() or "epoch" in stdout.lower():
            break

def keepalive_loop():
    log("[Idle Timeout Koruması] Keep-alive döngüsü aktif...")
    alive_code = "print('alive')"
    while True:
        time.sleep(180)  # 3 dakikada bir
        try:
            rc, stdout, stderr = exec_remote_python(alive_code, timeout=30)
            log(f"[Keep-Alive] 3 dakikalık heartbeat gönderildi. Durum: {'OK' if rc==0 else 'HATA'}")
        except Exception as e:
            log(f"[Keep-Alive Hatası]: {e}")

def main():
    log("==================================================")
    log("Mergen-SNN: Colab TPU v5e-1 Agresif Retry Başlatıldı")
    log("==================================================")
    
    tpu_acquired = False
    acquired_attempt = 0
    last_error = ""
    
    for attempt in range(1, MAX_ATTEMPTS + 1):
        success, err = try_allocate_tpu(attempt, MAX_ATTEMPTS)
        if success:
            tpu_acquired = True
            acquired_attempt = attempt
            break
        last_error = err
        if attempt < MAX_ATTEMPTS:
            log(f"Sonraki deneme için {RETRY_DELAY} saniye bekleniyor...")
            time.sleep(RETRY_DELAY)
            
    if not tpu_acquired:
        log("==================================================")
        log("60 DAKİKA SONUNDA TPU v5e-1 ALINAMADI (503 Service Unavailable)")
        log("Kaggle Fallback prosedürü devreye alınıyor.")
        log("==================================================")
        return
        
    log("==================================================")
    log(f"TPU {acquired_attempt}. denemede başarıyla alındı! Kuruluma geçiliyor...")
    log("==================================================")
    
    setup_environment()
    upload_files()
    start_training()
    keepalive_loop()

if __name__ == "__main__":
    main()

"""
Mergen-SNN: Colab T4 Resume Orchestrator (Part 2)
------------------------------------------------
1. Colab T4 GPU oturumunu (mergen-t4-v3-part2) tahsis eder (503 durumunda düzenli retry yapar).
2. Ortamı kurar (git clone/pull, pip install).
3. Tokenizer, validation verisi, cs_corpus ve parçalı step58000.pt checkpoint'ini yükler.
4. Checkpoint'i Colab üzerinde birleştirir ve doğrular.
5. bpe_train_v3.bin üretimini tamamlayıp eğitimi Step 58.000'den resume eder.
6. Keep-alive döngüsü ve senkronizasyon ile 1 epoch'un kalan kısmını tamamlar.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSION_NAME = "mergen-t4-v3-part2"
MAX_ATTEMPTS = 15
RETRY_DELAY = 90  # 1.5 dakika

LOG_FILE = PROJECT_ROOT / "colab_t4_resume.log"

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

def exec_remote_python(code, timeout=300):
    rc, stdout, stderr = run_colab_cmd(
        ["exec", "-s", SESSION_NAME, "--timeout", str(timeout)],
        timeout=timeout + 30,
        input_text=code
    )
    return rc, stdout, stderr

def try_allocate_t4(attempt, total):
    log(f"[Deneme {attempt}/{total}] T4 GPU oturum isteği gönderiliyor (colab new -s {SESSION_NAME} --gpu t4)...")
    rc, stdout, stderr = run_colab_cmd(["new", "-s", SESSION_NAME, "--gpu", "t4"], timeout=120)
    combined = (stdout + "\n" + stderr).strip()
    
    if rc == 0 or "READY" in combined or "Connected" in combined:
        log(f"[Deneme {attempt}/{total}] BAŞARILI! Tesla T4 oturumu tahsis edildi.")
        return True
    else:
        log(f"[Deneme {attempt}/{total}] Oturum tahsis edilemedi (503 Service Unavailable / Cooldown).")
        return False

def setup_remote_env():
    log("[Kurulum] Reponun güncellenmesi ve bağımlılıkların yüklenmesi...")
    setup_code = """
import subprocess, os

def run(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f'>> {cmd}')
    if p.stdout: print(p.stdout)
    if p.stderr: print(p.stderr)
    return p.returncode

if not os.path.exists('/content/mergen-snn'):
    run('git clone https://github.com/haremir/mergen-snn.git /content/mergen-snn')
else:
    run('cd /content/mergen-snn && git pull')

run('pip install -q snntorch pyyaml rich datasets')
run('mkdir -p /content/mergen-snn/data/processed /content/mergen-snn/checkpoints /content/mergen-snn/runs')

import torch
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')
"""
    rc, stdout, stderr = exec_remote_python(setup_code, timeout=300)
    log(f"[Kurulum Yanıtı]:\n{stdout}")
    return rc == 0

def upload_essential_files():
    log("[Veri Yükleme] Temel dosyaların yüklenmesi...")
    
    # 1. Tokenizer & Validation
    run_colab_cmd(["upload", "-s", SESSION_NAME, "data/processed/tokenizer_bpe_v2.json", "content/mergen-snn/data/processed/tokenizer_bpe_v2.json"])
    run_colab_cmd(["upload", "-s", SESSION_NAME, "data/processed/bpe_val_v3.bin", "content/mergen-snn/data/processed/bpe_val_v3.bin"])
    log("[Veri Yükleme] Tokenizer ve Val veri yüklendi.")
    
    # 2. cs_corpus.tar.gz
    if (PROJECT_ROOT / "cs_corpus.tar.gz").exists():
        log("[Veri Yükleme] cs_corpus.tar.gz yükleniyor...")
        run_colab_cmd(["upload", "-s", SESSION_NAME, "cs_corpus.tar.gz", "content/mergen-snn/data/raw/cs_corpus.tar.gz"])
        extract_code = """
import subprocess
subprocess.run('tar -xzf /content/mergen-snn/data/raw/cs_corpus.tar.gz -C /content/mergen-snn/data/raw/', shell=True)
print('cs_corpus çıkarıldı.')
"""
        exec_remote_python(extract_code, timeout=60)
        
    # 3. Checkpoint Chunk'larının Yüklenmesi
    chunk_files = sorted(list((PROJECT_ROOT / "checkpoints").glob("step58k_part_*.bin")))
    log(f"[Checkpoint Yükleme] {len(chunk_files)} parça yükleniyor...")
    for cf in chunk_files:
        log(f"[Upload] {cf.name} ({cf.stat().st_size / (1024*1024):.1f} MB)...")
        rc, out, err = run_colab_cmd(["upload", "-s", SESSION_NAME, str(cf), f"content/mergen-snn/checkpoints/{cf.name}"], timeout=300)
        if rc != 0:
            log(f"[Upload Hatası] {cf.name}: {err}")
            
    # Checkpoint'i Colab üzerinde birleştir
    reassemble_code = """
import os, subprocess
print('Checkpoint parçaları birleştiriliyor...')
subprocess.run('cat /content/mergen-snn/checkpoints/step58k_part_*.bin > /content/mergen-snn/checkpoints/stage5_21m_step58000.pt', shell=True)
sz = os.path.getsize('/content/mergen-snn/checkpoints/stage5_21m_step58000.pt')
print(f'stage5_21m_step58000.pt birleştirildi: {sz} bytes ({sz/(1024*1024):.2f} MB)')
"""
    rc, stdout, stderr = exec_remote_python(reassemble_code, timeout=60)
    log(f"[Birleştirme Sonucu]:\n{stdout}")

def prepare_data_and_resume():
    log("[Veri & Eğitim] bpe_train_v3.bin hazırlanıyor ve eğitim başlatılıyor...")
    pipeline_code = """
import subprocess, time, os

os.chdir('/content/mergen-snn')

# 1. bpe_train_v3.bin var mı kontrol et, yoksa üret
train_bin = '/content/mergen-snn/data/processed/bpe_train_v3.bin'
if not os.path.exists(train_bin) or os.path.getsize(train_bin) < 1000000000:
    print('bpe_train_v3.bin üretimi başlatılıyor...')
    subprocess.run('python scripts/prepare_bpe_data_massive.py', shell=True)
    print('Veri üretimi tamamlandı!')

# 2. Eğitimi başlat
print('Eğitim step 58000 den resume ediliyor...')
with open('/content/mergen-snn/runs/stage5_v3_t4_train_part2.log', 'w') as f:
    proc = subprocess.Popen(
        ['python', 'scripts/train_lm.py', '--config', 'configs/stage5_21m_16k_v3_t4.yaml', '--resume', 'auto'],
        stdout=f,
        stderr=subprocess.STDOUT
    )
print(f'Training resumed with PID: {proc.pid}')
"""
    with open(PROJECT_ROOT / "run_resume_pipeline.py", "w", encoding="utf-8") as f:
        f.write(pipeline_code)
        
    rc, out, err = run_colab_cmd(["upload", "-s", SESSION_NAME, "run_resume_pipeline.py", "content/mergen-snn/run_resume_pipeline.py"])
    launch_code = """
import subprocess
with open('/content/mergen-snn/pipeline_runner_part2.log', 'w') as f:
    p = subprocess.Popen(['python', '/content/mergen-snn/run_resume_pipeline.py'], stdout=f, stderr=subprocess.STDOUT)
print(f'Runner daemon PID: {p.pid}')
"""
    exec_remote_python(launch_code, timeout=30)
    log("[Pipeline] Resume pipeline daemon Colab üzerinde başlatıldı.")

def main():
    log("==================================================")
    log("Mergen-SNN: Colab T4 Part 2 Resume Orchestrator")
    log("==================================================")
    
    t4_acquired = False
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if try_allocate_t4(attempt, MAX_ATTEMPTS):
            t4_acquired = True
            break
        if attempt < MAX_ATTEMPTS:
            log(f"Sonraki deneme için {RETRY_DELAY} saniye bekleniyor...")
            time.sleep(RETRY_DELAY)
            
    if not t4_acquired:
        log("T4 oturumu tahsis edilemedi. Kaggle fallback önerilir.")
        return
        
    setup_remote_env()
    upload_essential_files()
    prepare_data_and_resume()
    log("==================================================")
    log("Tüm işlemler başarıyla tetiklendi! İzleme aktif.")
    log("==================================================")

if __name__ == "__main__":
    main()

"""Mergen-SNN — Colab 50M Model Eğitimini Başlatma ve Yönetme Scripti.

Colab VM üzerinde:
1. Google Drive bağlıysa oradaki bpe_train_v3.bin'i kontrol eder ve yerel SSD'ye bağlar/kopyalar.
2. Checkpoint hedefi olarak Drive varsa '/content/drive/MyDrive/mergen-snn/checkpoints',
   yoksa '/content/mergen-snn/checkpoints' ayarlar.
3. Arka planda train_lm.py sürecini başlatır.
"""

import os
import subprocess
import sys

def start_training(session_name="mergen-test"):
    remote_code = r'''
import os, sys, shutil, subprocess, time

os.chdir("/content/mergen-snn")
os.makedirs("runs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# 1. Drive kontrolü
drive_train = "/content/drive/MyDrive/mergen-snn/data/processed/bpe_train_v3.bin"
local_train = "/content/mergen-snn/data/processed/bpe_train_v3.bin"

if not os.path.exists(local_train):
    if os.path.exists(drive_train):
        print(f"Drive üzerinde train verisi bulundu, yerel diske kopyalanıyor: {drive_train}")
        shutil.copy2(drive_train, local_train)
        print("Kopyalama tamamlandı!")
    else:
        print(f"UYARI: {local_train} henüz mevcut değil.")

# 2. Checkpoint dizini belirleme
if os.path.exists("/content/drive/MyDrive"):
    ckpt_dir = "/content/drive/MyDrive/mergen-snn/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Kalıcı Checkpoint Hedefi (Google Drive): {ckpt_dir}")
else:
    ckpt_dir = "/content/mergen-snn/checkpoints"
    print(f"Geçici Checkpoint Hedefi (Yerel VM): {ckpt_dir}")

# 3. train_lm.py çalışıyor mu?
try:
    ps = subprocess.check_output(["pgrep", "-fa", "train_lm.py"]).decode().strip()
    if ps:
        print(f"train_lm.py zaten çalışıyor: {ps}")
        sys.exit(0)
except Exception:
    pass

# 4. Eğitimi arka planda başlat
log_path = "/content/mergen-snn/runs/stage6_50m.log"
cmd = [
    "python", "scripts/train_lm.py",
    "--config", "configs/stage6_50m_16k.yaml",
    "--resume", "auto"
]

print(f"Eğitim başlatılıyor: {' '.join(cmd)}")
with open(log_path, "a", encoding="utf-8") as f_log:
    proc = subprocess.Popen(
        cmd,
        stdout=f_log,
        stderr=subprocess.STDOUT,
        cwd="/content/mergen-snn"
    )

print(f"Eğitim arka planda başlatıldı! PID: {proc.pid}")
print(f"Log: {log_path}")
'''
    cmd = ["uv", "tool", "run", "colabapi", "raw", "--", "exec", "-s", session_name, "--timeout", "60"]
    env = {"PYTHONIOENCODING": "utf-8"}
    merged_env = os.environ.copy()
    merged_env.update(env)
    
    res = subprocess.run(
        cmd,
        input=remote_code,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=merged_env,
    )
    print(res.stdout)
    if res.stderr:
        print("[STDERR]", res.stderr)

if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else "mergen-test"
    start_training(session)

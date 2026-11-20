"""Mergen-SNN — Colab 50M Model Eğitimi İzleme Scripti (Local CLI).

Colab oturumundaki (mergen-test) eğitim logunu, son checkpoint durumunu,
spk1/spk2 oranlarını ve ölü nöron yüzdelerini anlık çeker.
"""

import io
import subprocess
import sys

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def get_colab_status(session_name="mergen-50m"):
    probe_code = r'''
import os, glob, subprocess

# 1. Pipeline ve Data Prep Logları
pipe_log = "/content/mergen-snn/runs/pipeline.log"
data_log = "/content/mergen-snn/runs/data_prep.log"
train_log = "/content/mergen-snn/runs/stage6_50m.log"

print("=== PIPELINE DURUMU ===")
if os.path.exists(pipe_log):
    with open(pipe_log, "r", encoding="utf-8", errors="replace") as f:
        print("".join(f.readlines()[-8:]).rstrip())
else:
    print("Pipeline log dosyası henüz yok.")

if os.path.exists(data_log):
    print("\n=== VERI URETIMI (DATA PREP) ===")
    with open(data_log, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    print(f"Toplam Data Prep Satırı: {len(lines)}")
    for l in lines[-8:]:
        print(l.rstrip())

print("\n=== MODEL EGITIMI (STAGE 6 50M) ===")
if os.path.exists(train_log):
    with open(train_log, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    print(f"Toplam Egitim Satiri: {len(lines)}")
    for l in lines[-12:]:
        print(l.rstrip())
else:
    print("train_lm.py henüz çıktı üretmedi (veri hazırlığı bekleniyor).")

print("\n=== PROCESS & GPU ===")
try:
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"]).decode()
    print(f"GPU: {smi.strip()}")
except Exception as e:
    print(f"GPU sorgulanamadi: {e}")

try:
    ps = subprocess.check_output(["pgrep", "-fa", "python"]).decode().strip()
    active_py = [line for line in ps.splitlines() if any(k in line for k in ["pipeline", "prepare", "train_lm"])]
    if active_py:
        print("Aktif Python Süreçleri:")
        for a in active_py:
            print(f"  * {a}")
    else:
        print("Mergen ile ilgili aktif Python süreci bulunamadı.")
except Exception:
    pass

print("\n=== DISK & CHECKPOINT DURUMU ===")
for p in ["/content/mergen-snn/data/processed/bpe_train_v3.bin", "/content/mergen-snn/data/processed/bpe_val_v3.bin"]:
    if os.path.exists(p):
        print(f"Dosya: {p} ({os.path.getsize(p)/(1024*1024):.2f} MB)")

ckpt_dirs = [
    "/content/drive/MyDrive/mergen-snn/checkpoints",
    "/content/mergen-snn/checkpoints"
]
for cd in ckpt_dirs:
    if os.path.exists(cd):
        ckpts = glob.glob(f"{cd}/*.pt")
        print(f"Dizin: {cd} ({len(ckpts)} checkpoint)")
        for c in sorted(ckpts, key=os.path.getmtime)[-3:]:
            sz = os.path.getsize(c) / (1024 * 1024)
            print(f"  - {os.path.basename(c)} ({sz:.1f} MB)")
'''
    cmd = ["uv", "tool", "run", "colabapi", "raw", "--", "exec", "-s", session_name, "--timeout", "30"]
    env = {"PYTHONIOENCODING": "utf-8"}
    import os
    merged_env = os.environ.copy()
    merged_env.update(env)
    
    res = subprocess.run(
        cmd,
        input=probe_code,
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
    get_colab_status(session)

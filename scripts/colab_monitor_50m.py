"""Mergen-SNN — Colab 50M Model Eğitimi İzleme Scripti (Local CLI).

Colab oturumundaki (mergen-test) eğitim logunu, son checkpoint durumunu,
spk1/spk2 oranlarını ve ölü nöron yüzdelerini anlık çeker.
"""

import subprocess
import sys

def get_colab_status(session_name="mergen-test"):
    probe_code = r'''
import os, glob, subprocess

log_file = "/content/mergen-snn/runs/stage6_50m.log"
print("=== LOG DURUMU ===")
if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    print(f"Toplam Log Satiri: {len(lines)}")
    print("--- Son 15 Satir ---")
    for l in lines[-15:]:
        print(l.rstrip())
else:
    print("Henuz log dosyasi olusmadi.")

print("\n=== GPU VE PROCESS ===")
try:
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"]).decode()
    print(f"GPU: {smi.strip()}")
except Exception as e:
    print(f"GPU sorgulanamadi: {e}")

try:
    ps = subprocess.check_output(["pgrep", "-fa", "train_lm.py"]).decode().strip()
    if ps:
        print(f"Egitim Sureci Aktif: {ps}")
    else:
        print("train_lm.py su anda calismiyor.")
except Exception:
    print("train_lm.py calismiyor.")

print("\n=== CHECKPOINT DURUMU ===")
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

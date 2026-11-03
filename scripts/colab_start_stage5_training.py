import subprocess
import os
import sys

os.chdir("/content/mergen-snn")
os.makedirs("/content/mergen-snn/checkpoints", exist_ok=True)
os.makedirs("/content/mergen-snn/runs", exist_ok=True)

log_fp = open("/content/mergen-snn/train.log", "w", encoding="utf-8")

cmd = [
    sys.executable,
    "scripts/train_lm.py",
    "--config",
    "configs/stage5_21m_16k_tpu.yaml",
    "--resume",
    "auto"
]

print("Launching Stage 5 training on TPU:")
print(" ".join(cmd))

p = subprocess.Popen(
    cmd,
    stdout=log_fp,
    stderr=subprocess.STDOUT,
    cwd="/content/mergen-snn"
)

print(f"Training successfully launched in background with PID {p.pid}!")

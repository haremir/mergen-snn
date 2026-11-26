"""Start Stage 8 100M Hybrid LIF training on Colab session in the background."""

import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from colab_exec import exec_on_colab

SESSION = "mergen-stage8-100m"


def main():
    print(f"Starting Stage 8 training on Colab session '{SESSION}'...")

    remote_start_code = r'''
import os, sys, subprocess

repo_dir = "/content/mergen-snn"
os.chdir(repo_dir)
os.makedirs("runs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# 1. Kontrol: train_lm.py zaten calisiyor mu?
try:
    ps = subprocess.check_output(["pgrep", "-fa", "train_lm.py"]).decode().strip()
    if ps:
        print("ALREADY_RUNNING:", ps)
        sys.exit(0)
except Exception:
    pass

# 2. Arka planda baslat
log_path = "/content/mergen-snn/runs/stage8_hybrid_100m.log"
cmd = [
    sys.executable, "scripts/train_lm.py",
    "--config", "configs/stage8_hybrid_100m.yaml",
    "--resume", "auto"
]

env = os.environ.copy()
env["PYTHONPATH"] = "/content/mergen-snn/src:" + env.get("PYTHONPATH", "")
env["PYTHONUNBUFFERED"] = "1"

with open(log_path, "a", encoding="utf-8") as log_file:
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=repo_dir,
        env=env,
        start_new_session=True
    )

print(f"STARTED_PID:{proc.pid}")
'''
    res = exec_on_colab(remote_start_code, session=SESSION, timeout=30)
    print("Colab response:", res.stdout.strip())
    if res.stderr:
        print("Colab stderr:", res.stderr.strip())

    time.sleep(4)
    status_code = r'''
import os, subprocess
log_path = "/content/mergen-snn/runs/stage8_hybrid_100m.log"
try:
    ps = subprocess.check_output(["pgrep", "-fa", "train_lm.py"]).decode().strip()
except Exception:
    ps = "NOT_FOUND"

log_tail = ""
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        log_tail = "".join(f.readlines()[-12:]).strip()

print(f"PS:{ps}\nLOG:\n{log_tail}")
'''
    res2 = exec_on_colab(status_code, session=SESSION, timeout=30)
    print("\n--- Training Launch Status ---")
    print(res2.stdout.strip())


if __name__ == "__main__":
    main()

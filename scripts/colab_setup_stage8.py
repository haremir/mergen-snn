"""Setup Colab environment for Stage 8."""

import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from colab_exec import exec_on_colab

SESSION = "mergen-stage8-100m"

COLAB_SCRIPT = r"""
import os, subprocess, shutil

print("--- CLONING MERGEN-SNN REPO ---")
os.chdir("/content")
if os.path.exists("/content/mergen-snn"):
    shutil.rmtree("/content/mergen-snn", ignore_errors=True)

subprocess.run(["git", "clone", "https://github.com/haremir/mergen-snn.git"], check=True)

print("--- INSTALLING DEPENDENCIES ---")
subprocess.run([
    "pip", "install", "-q",
    "snntorch", "pyyaml", "rich", "numpy", "datasets", "tokenizers"
], check=True)

subprocess.run([
    "pip", "install", "-q", "-e", "/content/mergen-snn", "--no-deps"
], check=True)

for d in ["/content/mergen-snn/data/processed", "/content/mergen-snn/checkpoints", "/content/mergen-snn/runs"]:
    os.makedirs(d, exist_ok=True)

print("--- VERIFYING GIT LOG ---")
res = subprocess.check_output(["git", "-C", "/content/mergen-snn", "log", "-1", "--oneline"]).decode()
print("Latest commit:", res.strip())
print("SETUP COMPLETED SUCCESSFULLY!")
"""

def main():
    print(f"Setting up {SESSION}...")
    res = exec_on_colab(COLAB_SCRIPT, session=SESSION, timeout=240)
    print("STDOUT:")
    print(res.stdout)
    if res.stderr:
        print("STDERR:")
        print(res.stderr)
    return res.returncode

if __name__ == "__main__":
    sys.exit(main())

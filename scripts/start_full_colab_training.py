import os
import subprocess
import sys
import time

repo_dir = "/content/mergen-snn"

# 1. Kill any existing train_lm process
print("Checking for existing training processes...")
subprocess.run(["pkill", "-9", "-f", "train_lm.py"])
time.sleep(1)

# 2. Clean test checkpoints
ckpt_dir = os.path.join(repo_dir, "checkpoints")
for f in os.listdir(ckpt_dir):
    if f.endswith(".pt") or f.endswith(".png"):
        os.remove(os.path.join(ckpt_dir, f))
print(f"Cleaned test checkpoints in {ckpt_dir}")

# 3. Clear/create train.log
log_path = os.path.join(repo_dir, "train.log")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("=== Mergen-SNN Stage 4 Training Log Started ===\n")

# 4. Launch full training in background
env = os.environ.copy()
env["PYTHONPATH"] = f"{repo_dir}/src"

cmd = [
    "python3",
    "scripts/train_lm.py",
    "--config", "configs/stage4_small.yaml",
    "--resume", "auto"
]

print(f"Launching full training on Tesla T4: {' '.join(cmd)}")
with open(log_path, "a", encoding="utf-8") as out:
    proc = subprocess.Popen(
        cmd,
        cwd=repo_dir,
        env=env,
        stdout=out,
        stderr=subprocess.STDOUT,
        start_new_session=True
    )

print(f"Training successfully spawned in background! PID = {proc.pid}")

# 5. Wait 6 seconds and display initial log
time.sleep(6)
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        print("\n=== Initial train.log ===")
        print(f.read())
        print("=========================")

import os
import subprocess
import sys
import time

print("Checking remote environment...")
repo_dir = "/content/mergen-snn"

# 1. Check data files
train_bin = os.path.join(repo_dir, "data/processed/bpe_train.bin")
val_bin = os.path.join(repo_dir, "data/processed/bpe_val.bin")
tok_json = os.path.join(repo_dir, "data/processed/tokenizer_bpe.json")

for path in [train_bin, val_bin, tok_json]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"File exists: {path} ({size_mb:.2f} MB)")
    else:
        print(f"ERROR: Missing file {path}")
        sys.exit(1)

# 2. Check if git repo is updated
subprocess.run(["git", "pull"], cwd=repo_dir, check=False)

# 3. Check if already running
check_ps = subprocess.run(["pgrep", "-f", "train_lm.py"], capture_output=True, text=True)
if check_ps.stdout.strip():
    print(f"Warning: train_lm.py is already running with PID(s): {check_ps.stdout.strip()}")
else:
    # 4. Start training in background
    log_path = os.path.join(repo_dir, "train.log")
    print(f"Starting training in background, logs will be written to {log_path}...")
    
    with open(log_path, "a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            ["uv", "run", "python", "scripts/train_lm.py", "--config", "configs/stage4_small.yaml", "--resume", "auto"],
            cwd=repo_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
    print(f"Training started successfully with PID: {proc.pid}")

# Wait 3 seconds and print initial log lines
time.sleep(3)
log_path = os.path.join(repo_dir, "train.log")
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        print("\n--- Initial train.log output ---")
        print(f.read())
        print("--- End of initial train.log ---")

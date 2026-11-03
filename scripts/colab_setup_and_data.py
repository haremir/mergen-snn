import os
import sys
import shutil
import subprocess

print("=== STEP 1: CLONE OR UPDATE REPOSITORY ===")
if not os.path.exists("/content/mergen-snn"):
    print("Cloning mergen-snn repository...")
    subprocess.run(["git", "clone", "https://github.com/haremir/mergen-snn.git", "/content/mergen-snn"], check=True)
else:
    print("Updating existing mergen-snn repository...")
    subprocess.run(["git", "-C", "/content/mergen-snn", "pull"], check=True)

print("=== STEP 2: INSTALL DEPENDENCIES ===")
pkgs = ["snntorch", "pyyaml", "rich", "datasets", "tokenizers"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
print("Dependencies installed successfully.")

print("=== STEP 3: SETUP DATA PROCESSED DIRECTORY ===")
data_dir = "/content/mergen-snn/data/processed"
os.makedirs(data_dir, exist_ok=True)

for fname in ["tokenizer_bpe_v2.json", "bpe_val_v2.bin"]:
    src = os.path.join("/content", fname)
    dst = os.path.join(data_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {fname} ({os.path.getsize(dst)/(1024*1024):.2f} MB) to {data_dir}")
    elif os.path.exists(dst):
        print(f"File {fname} already exists in {data_dir} ({os.path.getsize(dst)/(1024*1024):.2f} MB)")

print("=== STEP 4: VERIFY DATA PROCESSED DIRECTORY ===")
for f in os.listdir(data_dir):
    fp = os.path.join(data_dir, f)
    print(f" - {f}: {os.path.getsize(fp)/(1024*1024):.2f} MB")

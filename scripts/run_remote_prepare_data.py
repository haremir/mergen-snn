import subprocess
import sys
import os

os.chdir("/content/mergen-snn")
print("Current dir:", os.getcwd())

cmd = [sys.executable, "scripts/prepare_bpe_data_massive.py"]
print("Running:", " ".join(cmd))
res = subprocess.run(cmd)
print("Exit code:", res.returncode)

data_dir = "/content/mergen-snn/data/processed"
for f in os.listdir(data_dir):
    fp = os.path.join(data_dir, f)
    print(f" - {f}: {os.path.getsize(fp)/(1024*1024):.2f} MB")

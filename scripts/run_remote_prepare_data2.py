import subprocess
import sys
import os

os.chdir("/content/mergen-snn")
cmd = [sys.executable, "scripts/prepare_bpe_data_massive.py"]
res = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
print("STDERR:")
print(res.stderr)

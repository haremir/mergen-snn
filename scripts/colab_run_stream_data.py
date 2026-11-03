import os
import sys
import subprocess

os.chdir("/content/mergen-snn")
print("Starting prepare_bpe_data_massive.py on Colab...")

# Run and print output in real-time or log to file
with open("/content/mergen-snn/data_prep.log", "w") as logf:
    p = subprocess.Popen([sys.executable, "scripts/prepare_bpe_data_massive.py"], stdout=logf, stderr=subprocess.STDOUT)
    print(f"Data preparation started with PID {p.pid}")

import os
import subprocess

print("--- PROCESS CHECK ---")
ps = subprocess.run(["pgrep", "-fa", "prepare_bpe_data_massive"], capture_output=True, text=True)
print(ps.stdout)

print("--- LOG FILE (last 25 lines) ---")
if os.path.exists("/content/mergen-snn/data_prep.log"):
    tail = subprocess.run(["tail", "-n", "25", "/content/mergen-snn/data_prep.log"], capture_output=True, text=True)
    print(tail.stdout)

print("--- BINARY FILE SIZES ---")
data_dir = "/content/mergen-snn/data/processed"
if os.path.exists(data_dir):
    for f in os.listdir(data_dir):
        fp = os.path.join(data_dir, f)
        print(f" - {f}: {os.path.getsize(fp)/(1024*1024):.2f} MB")

import os
import signal
import subprocess
import time

ps = subprocess.run(["pgrep", "-f", "prepare_bpe_data_massive"], capture_output=True, text=True)
pids = [int(p) for p in ps.stdout.strip().split() if p.isdigit()]
print("Killing data prep processes:", pids)
for pid in pids:
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception as e:
        print("Kill error:", e)

time.sleep(2)

data_dir = "/content/mergen-snn/data/processed"
for f in os.listdir(data_dir):
    fp = os.path.join(data_dir, f)
    print(f" - {f}: {os.path.getsize(fp)/(1024*1024):.2f} MB")

import os
import subprocess

print("--- PROCESS STATUS ---")
ps = subprocess.run(["pgrep", "-fa", "train_lm.py"], capture_output=True, text=True)
print(ps.stdout)

print("--- TRAIN LOG (last 40 lines) ---")
if os.path.exists("/content/mergen-snn/train.log"):
    tail = subprocess.run(["tail", "-n", "40", "/content/mergen-snn/train.log"], capture_output=True, text=True)
    print(tail.stdout)
else:
    print("train.log not found yet.")

print("--- CHECKPOINTS ---")
ckpt_dir = "/content/mergen-snn/checkpoints"
if os.path.exists(ckpt_dir):
    print("Checkpoints:", os.listdir(ckpt_dir))

import os
import subprocess

print("=== Remote Process Status ===")
res = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
train_procs = [line for line in res.stdout.splitlines() if "train_lm.py" in line]
if train_procs:
    for line in train_procs:
        print("Running:", line)
else:
    print("No train_lm.py process running!")

print("\n=== NVIDIA GPU Status ===")
gpu_res = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True)
print(gpu_res.stdout.strip())

print("\n=== Recent Checkpoints ===")
ckpt_dir = "/content/mergen-snn/checkpoints"
if os.path.exists(ckpt_dir):
    ckpts = os.listdir(ckpt_dir)
    print(f"Checkpoints ({len(ckpts)} files):", ckpts[-5:] if ckpts else "None")
else:
    print("Checkpoint directory does not exist yet.")

print("\n=== Tail of train.log ===")
log_path = "/content/mergen-snn/train.log"
if os.path.exists(log_path):
    tail_res = subprocess.run(["tail", "-n", "35", log_path], capture_output=True, text=True)
    print(tail_res.stdout)
else:
    print("train.log not found.")

import sys
import subprocess

print("--- PIP LIST (XLA / TPU / TORCH) ---")
res = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
for line in res.stdout.splitlines():
    if any(k in line.lower() for k in ["torch", "xla", "tpu", "jax"]):
        print(line)

print("--- TESTING TORCH TPU DEVICE ---")
import torch
print("torch.__version__:", torch.__version__)
print("is_built_with_xla:", hasattr(torch, "xla"))

try:
    d = torch.device("tpu")
    print("torch.device('tpu'):", d)
    t = torch.ones((2, 2), device=d)
    print("t on TPU:", t)
except Exception as e:
    print("torch.device('tpu') failed:", e)

try:
    import torch_xla
    print("torch_xla found!")
except Exception as e:
    print("import torch_xla failed:", e)

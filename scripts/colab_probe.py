import os
import subprocess
import torch

print("=== Colab Remote Probe ===")
print("Python executable:", os.sys.executable)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
    print("GPU Count:", torch.cuda.device_count())

print("\n--- nvidia-smi ---")
try:
    res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(res.stdout)
except Exception as e:
    print("nvidia-smi error:", e)

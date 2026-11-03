import os
import subprocess

print("Killing current CPU train processes...")
subprocess.run(["pkill", "-9", "-f", "train_lm.py"])

print("Checking Colab system torch vs venv torch:")
# System Python
try:
    res = subprocess.run(["python3", "-c", "import torch; print('System torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"], capture_output=True, text=True)
    print(res.stdout)
except Exception as e:
    print("System error:", e)

# Venv Python
try:
    res = subprocess.run(["/content/mergen-snn/.venv/bin/python", "-c", "import torch; print('Venv torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"], capture_output=True, text=True)
    print(res.stdout)
except Exception as e:
    print("Venv error:", e)

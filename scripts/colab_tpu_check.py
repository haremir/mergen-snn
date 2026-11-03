import os
import sys

print(f"Python: {sys.version}")
print(f"OS: {sys.platform}")

os.environ['PJRT_DEVICE'] = 'TPU'

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except Exception as e:
    print(f"PyTorch import failed: {e}")

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print(f"TPU cihazı: {device}")
    print("torch_xla başarıyla yüklendi.")
except Exception as e:
    print(f"TPU başlatılamadı: {e}")
    print("GPU'ya (T4) geri dönülüyor...")

try:
    import snntorch
    print(f"snntorch version: {snntorch.__version__}")
except Exception as e:
    print(f"snntorch not installed or failed: {e}")

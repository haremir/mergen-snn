import subprocess

print("Checking missing packages in system python:")
code = """
import torch
print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
for pkg in ['snntorch', 'tokenizers', 'yaml', 'tqdm']:
    try:
        __import__(pkg)
        print(f'{pkg}: INSTALLED')
    except ImportError:
        print(f'{pkg}: MISSING')
"""
res = subprocess.run(["python3", "-c", code], capture_output=True, text=True)
print(res.stdout)

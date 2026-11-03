import subprocess
import sys

repo_dir = "/content/mergen-snn"
print("Installing torch+cu124 into venv...")

cmd = [
    "uv", "pip", "install", "--upgrade",
    "torch", "torchvision",
    "--index-url", "https://download.pytorch.org/whl/cu124"
]
res = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
print("Output:", res.stdout[-500:] if len(res.stdout) > 500 else res.stdout)
if res.stderr:
    print("Stderr:", res.stderr[-500:] if len(res.stderr) > 500 else res.stderr)

# Verify
test_code = "import torch; print('CUDA available in venv:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
res_test = subprocess.run(["/content/mergen-snn/.venv/bin/python", "-c", test_code], capture_output=True, text=True)
print("Test result:", res_test.stdout.strip())

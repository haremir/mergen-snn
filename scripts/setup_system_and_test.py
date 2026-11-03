import subprocess
import sys

print("Installing snntorch and mergen dependencies in system python...")
res = subprocess.run(["pip", "install", "snntorch"], capture_output=True, text=True)
print(res.stdout[-300:] if len(res.stdout) > 300 else res.stdout)
if res.stderr:
    print(res.stderr[-300:] if len(res.stderr) > 300 else res.stderr)

# Also install mergen package in editable mode
res_pkg = subprocess.run(["pip", "install", "-e", ".", "--no-deps"], cwd="/content/mergen-snn", capture_output=True, text=True)
print("Editable install:", res_pkg.stdout)

# Test CUDA device and forward pass
test_code = """
import torch
import snntorch as snn
from mergen.models.bpe_snn_lm import BpeSNNLM
from mergen.utils.device import get_device

device = get_device("auto")
print(f"Detected Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

model = BpeSNNLM(vocab_size=8192, embed_dim=512, hidden_size=512, num_steps=32, tie_weights=False, use_cython=False).to(device)
dummy_input = torch.randint(0, 8192, (2, 32), device=device)
logits, stats = model(dummy_input)
print(f"Forward pass success! Logits shape: {logits.shape}, LIF2 spike ratio: {stats['spike_ratio_lif2']:.4f}")
"""

res_test = subprocess.run(["python3", "-c", test_code], cwd="/content/mergen-snn", capture_output=True, text=True)
print("=== Forward Test Result ===")
print(res_test.stdout)
if res_test.stderr:
    print("Stderr:", res_test.stderr)

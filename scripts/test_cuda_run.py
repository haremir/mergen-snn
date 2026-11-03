import os
import sys

# Add src to path
sys.path.insert(0, "/content/mergen-snn/src")

import torch
import snntorch as snn
from mergen.models.snn_lm_bpe import BpeSNNLM
from mergen.utils.device import get_device

device = get_device("auto")
print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

model = BpeSNNLM(
    vocab_size=8192,
    embed_dim=512,
    hidden_size=512,
    tie_weights=False,
    use_cython=False
).to(device)

dummy_input = torch.randint(0, 8192, (2, 32), device=device)
logits, stats = model(dummy_input)
print(f"Forward pass SUCCESS! Logits shape: {logits.shape}, LIF2 spike ratio: {stats['spike_ratio_lif2']:.4f}")
print("CUDA Memory Allocated:", round(torch.cuda.memory_allocated() / (1024 * 1024), 2), "MB")

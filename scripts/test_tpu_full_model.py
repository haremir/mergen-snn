import os
import sys
import subprocess
import torch
import torch.nn as nn

print("=== VERIFYING REPOSITORY AND IMPORTING ACTUAL BpeSNNLM ===")
if not os.path.exists("mergen-snn"):
    print("Cloning mergen-snn repo...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/haremir/mergen-snn.git"], check=True)
else:
    print("mergen-snn repo already exists. Pulling latest...")
    subprocess.run(["git", "-C", "mergen-snn", "pull"], check=False)

sys.path.insert(0, os.path.abspath("mergen-snn/src"))

from mergen.models.snn_lm_bpe import BpeSNNLM

device = torch.device("tpu")
print(f"Device: {device}")

print("Initializing BpeSNNLM(vocab_size=100, embed_dim=32, hidden_size=32, beta=0.9, use_cython=False, tie_weights=True)...")
model = BpeSNNLM(
    vocab_size=100,
    embed_dim=32,
    hidden_size=32,
    beta=0.9,
    use_cython=False,
    tie_weights=True
)
print("Model parameter count:", sum(p.numel() for p in model.parameters()))

print(f"Moving model to {device}...")
model = model.to(device)

inputs = torch.randint(0, 100, (2, 4), device=device)
targets = torch.randint(0, 100, (2, 4), device=device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Executing forward pass with BpeSNNLM...")
logits = model(inputs, num_steps=4)
print(f"Forward success! Logits shape: {logits.shape}, device: {logits.device}")

print("Executing loss calculation...")
loss = criterion(logits.view(-1, 100), targets.view(-1))
print(f"Loss: {loss.item():.4f}")

print("Executing backward pass...")
optimizer.zero_grad()
loss.backward()
print("Backward success!")

print("Executing optimizer.step()...")
optimizer.step()
print("Optimizer step success!")

print("\n>>> EXACT BpeSNNLM CLASS PASSED ALL TESTS ON TPU v5e-1! <<<")

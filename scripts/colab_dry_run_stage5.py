import os
import sys
import torch
import yaml

os.chdir("/content/mergen-snn")
sys.path.insert(0, "/content/mergen-snn/src")

from mergen.models.snn_lm_bpe import BpeSNNLM
from mergen.data.char_dataset import CharDataset
from mergen.utils.device import get_device, device_name

device = get_device("tpu")
print(f"Device: {device_name(device)} ({device})")

train_bin = "data/processed/bpe_train_v2.bin"
val_bin = "data/processed/bpe_val_v2.bin"
print("Loading train dataset from", train_bin)
train_ds = CharDataset(train_bin, block_size=128, stride=128)
print(f"Train dataset length: {len(train_ds):,} samples")

val_ds = CharDataset(val_bin, block_size=128, stride=128)
print(f"Val dataset length: {len(val_ds):,} samples")

print("Initializing 21M BpeSNNLM...")
model = BpeSNNLM(
    vocab_size=16384,
    embed_dim=640,
    hidden_size=640,
    beta=0.9,
    tie_weights=False,
    use_cython=False
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"TOTAL MODEL PARAMETERS: {total_params:,}")

# Test 1 batch forward & backward on TPU
x, y = train_ds[0]
x = x.unsqueeze(0).repeat(64, 1).to(device)  # batch 64, block_size 128
y = y.unsqueeze(0).repeat(64, 1).to(device)

print(f"Inputs on TPU: {x.shape}, {x.device}")
print("Running forward pass (num_steps=32)...")
logits, spk1, spk2 = model(x, num_steps=32, return_spikes=True)
spk_ratio = (spk2 > 0).float().mean().item()
print(f"Forward success! Logits shape: {logits.shape}, spk2_ratio: {spk_ratio:.4f}")

criterion = torch.nn.CrossEntropyLoss()
loss = criterion(logits.reshape(-1, logits.size(-1)), y[:, :32].reshape(-1))
print(f"Initial CrossEntropyLoss: {loss.item():.4f} (expected ~ln(16384) = 9.704)")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
optimizer.zero_grad()
loss.backward()
print("Backward pass success!")
optimizer.step()
print("Optimizer step success!")
print(">>> DRY RUN COMPLETE AND 100% VERIFIED ON TPU! <<<")

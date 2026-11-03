import sys
import subprocess
import os

print("=== STEP 1: INSTALL SNNTORCH ===")
try:
    import snntorch
    print(f"snntorch already installed: {snntorch.__version__}")
except ImportError:
    print("Installing snntorch...")
    res = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "snntorch"], capture_output=True, text=True)
    if res.returncode != 0:
        print("pip install snntorch failed:", res.stderr)
    else:
        print("snntorch installed successfully.")
        import snntorch
        print(f"snntorch version: {snntorch.__version__}")

print("\n=== STEP 2: DEVICE DETECTION ===")
import torch
print(f"PyTorch version: {torch.__version__}")

device = None
# Check torch_xla
try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print(f"torch_xla device: {device}")
except Exception as e:
    print(f"torch_xla not available: {e}")

# Check native torch_tpu
if device is None:
    try:
        d = torch.device("tpu")
        t = torch.ones((1,), device=d)
        device = d
        print(f"torch native TPU device verified: {device} (tensor: {t})")
    except Exception as e:
        print(f"torch native TPU failed: {e}")

print(f"\nFinal target device: {device}")

print("\n=== STEP 3: MINI SNN COMPATIBILITY TEST ===")
if device is None:
    print("TPU device is NOT available! Cannot run SNN test.")
    sys.exit(1)

# Clone or inject BpeSNNLM definition directly to avoid path issues
import torch.nn as nn
import snntorch as snn

class MiniBpeSNNLM(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=32, hidden_size=32, beta=0.9):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.beta = beta
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        emb = self.embedding(input_ids)
        cur1 = self.fc1(emb)
        
        mem1 = self.lif1.init_leaky()
        spk1_list = []
        for t in range(seq_len):
            spk1, mem1 = self.lif1(cur1[:, t, :], mem1)
            spk1_list.append(spk1)
        spk1_all = torch.stack(spk1_list, dim=1)
        
        cur2 = self.fc2(spk1_all)
        mem2 = self.lif2.init_leaky()
        spk2_list = []
        for t in range(seq_len):
            spk2, mem2 = self.lif2(cur2[:, t, :], mem2)
            spk2_list.append(spk2)
        spk2_all = torch.stack(spk2_list, dim=1)
        
        logits = self.fc_out(spk2_all)
        return logits

try:
    print("Creating MiniBpeSNNLM...")
    model = MiniBpeSNNLM(vocab_size=100, embed_dim=32, hidden_size=32, beta=0.9)
    print(f"Moving model to {device}...")
    model = model.to(device)
    print("Model moved successfully.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Input batch=2, seq=4
    inputs = torch.randint(0, 100, (2, 4), device=device)
    targets = torch.randint(0, 100, (2, 4), device=device)
    print(f"Inputs created on {inputs.device}: shape {inputs.shape}")
    
    print("Running forward pass...")
    logits = model(inputs)
    print(f"Forward pass completed! Logits shape: {logits.shape}, device: {logits.device}")
    
    print("Computing loss...")
    loss = criterion(logits.view(-1, 100), targets.view(-1))
    print(f"Loss computed: {loss.item()}")
    
    print("Running backward pass...")
    optimizer.zero_grad()
    loss.backward()
    print("Backward pass completed!")
    
    print("Running optimizer.step()...")
    optimizer.step()
    print("optimizer.step() completed successfully!")
    print("\n>>> ALL TESTS PASSED SUCCESSFULLY ON TPU! <<<")

except Exception as e:
    import traceback
    print("\n>>> TPU SNN TEST FAILED! <<<")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

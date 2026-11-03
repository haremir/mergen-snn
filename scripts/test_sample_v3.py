import torch
from mergen.models.snn_lm_bpe import BpeSNNLM
from tokenizers import Tokenizer

device = torch.device("cpu")
tokenizer = Tokenizer.from_file("data/processed/tokenizer_bpe_v2.json")
ckpt = torch.load("checkpoints/stage5_21m_step58000.pt", map_location="cpu", weights_only=False)

model = BpeSNNLM(
    vocab_size=16384,
    embed_dim=640,
    hidden_size=640,
    beta=0.9,
    tie_weights=False,
    use_cython=False
).to(device)

model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

def generate(prompt, max_tokens=35):
    tokens = tokenizer.encode(prompt).ids
    for _ in range(max_tokens):
        x = torch.tensor([tokens[-128:]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(x, num_steps=8)
            logits = out[0] if isinstance(out, tuple) else out
            next_token = torch.argmax(logits[0, -1, :]).item()
            tokens.append(next_token)
    return tokenizer.decode(tokens)

for p in ["def train(", "Python'da liste", "Algoritma nedir?"]:
    print(f"\n--- PROMPT: {p} ---")
    print(generate(p))

"""Test text generation with Stage 7 Spike Transformer 10M models."""

import os
from pathlib import Path
import sys
import torch
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from mergen.models.spike_transformer import SpikeTransformerLM


def sample_token(logits, temperature=0.8, top_k=40, top_p=0.9):
    temp = max(float(temperature), 1e-4)
    filtered = logits.clone() / temp
    if top_k > 0:
        val, _ = torch.topk(filtered, min(top_k, filtered.size(-1)))
        filtered[filtered < val[-1]] = -float("Inf")
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        filtered[indices_to_remove] = -float("Inf")
    probs = torch.softmax(filtered, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate(model, tokenizer, prompt, max_tokens=60, temperature=0.8, top_k=40, top_p=0.9, block_size=128):
    model.eval()
    encoded = tokenizer.encode(prompt).ids
    current_ids = list(encoded)
    pad_id = tokenizer.token_to_id("<pad>") or 0

    with torch.no_grad():
        for _ in range(max_tokens):
            if len(current_ids) < block_size:
                pad_len = block_size - len(current_ids)
                window_ids = [pad_id] * pad_len + current_ids
            else:
                window_ids = current_ids[-block_size:]

            inp = torch.tensor([window_ids], dtype=torch.long)
            out = model(inp)
            logits = out[0] if isinstance(out, tuple) else out
            last_logits = logits[0, -1, :]

            next_id = sample_token(last_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            current_ids.append(next_id)

    return tokenizer.decode(current_ids)


def main():
    tok_path = PROJECT_ROOT / "data" / "processed" / "tokenizer_bpe_v2.json"
    if not tok_path.exists():
        tok_path = PROJECT_ROOT / "data" / "processed" / "tokenizer_bpe.json"
    tokenizer = Tokenizer.from_file(str(tok_path))

    model = SpikeTransformerLM(
        vocab_size=16384,
        embed_dim=256,
        num_blocks=4,
        num_heads=4,
        ffn_hidden=512,
        max_seq=128,
        beta=0.9,
        threshold=0.5,
    )

    ckpts = [
        ("Epoch 8 (Final)", PROJECT_ROOT / "checkpoints" / "stage7_spike_transformer_10m_epoch8_weights.pt"),
        ("Best Model (Val 7.792)", PROJECT_ROOT / "checkpoints" / "stage7_spike_transformer_10m_best.pt"),
    ]

    prompts = [
        "Python'da liste",
        "def train(",
        "Algoritma nedir?",
        "Yapay zeka",
    ]

    for label, ckpt_path in ckpts:
        if not ckpt_path.exists():
            print(f"Skipping {ckpt_path.name} (not found)")
            continue
        print(f"\n==========================================")
        print(f"--- MODEL: {label} ({ckpt_path.name}) ---")
        print(f"==========================================")
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = data["model_state_dict"] if "model_state_dict" in data else data
        model.load_state_dict(sd, strict=False)

        for p in prompts:
            out = generate(model, tokenizer, p, max_tokens=50, temperature=0.7, top_k=30, top_p=0.85)
            print(f"\n[PROMPT]: {p}")
            print(f"[ÇIKTI]:\n{out.strip()}\n" + "-"*40)


if __name__ == "__main__":
    main()

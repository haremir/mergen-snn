"""Mergen-SNN — Chat Model Test Çıkarımı."""

import math
from pathlib import Path
import sys
import time

import torch
from tokenizers import Tokenizer

from mergen.models.snn_lm_bpe import BpeSNNLM

# Windows Unicode konsol uyumluluğu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SYSTEM_PROMPT = "Sen Mergen'sin. Bilgisayar bilimi uzmanısın. CS dışı konularda kibarca reddet."
PROMPTS = [
    "Merhaba",
    "Python'da liste nedir?",
    "Hava nasıl?",
    "def train(",
]


def test_inference(checkpoint_path: Path = Path("checkpoints/mergen_sft.pt")):
    tokenizer = Tokenizer.from_file("data/processed/tokenizer_bpe.json")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = BpeSNNLM(
        vocab_size=checkpoint.get("vocab_size", 8196),
        embed_dim=checkpoint.get("embed_dim", 384),
        hidden_size=checkpoint.get("hidden_size", 384),
        beta=checkpoint.get("beta", 0.9),
        tie_weights=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("=" * 70)
    print("MERGEN-SNN CHAT TEST ÇIKARIMI (4 ÖRNEK)")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 70)

    for i, prompt in enumerate(PROMPTS, 1):
        formatted = f"<|system|>{SYSTEM_PROMPT}<|end|><|user|>{prompt}<|end|><|assistant|>"
        enc = tokenizer.encode(formatted)

        t0 = time.time()
        output_ids, metrics = model.generate_with_metrics(
            enc,
            max_new_tokens=60,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            end_token_id=8195,
        )
        elapsed_ms = (time.time() - t0) * 1000

        response = tokenizer.decode(output_ids).strip()

        print(f"\n[Örnek {i}]")
        print(f"Kullanıcı Mesajı (Prompt): {prompt!r}")
        print(f"Asistan Yanıtı (Response):\n{response}")
        print("\nMetrikler:")
        print(
            f"⏱️ Süre: {elapsed_ms:.0f}ms | 📊 Loss: {metrics['loss']:.3f} | PPL: {metrics['ppl']:.1f} | BPB: {metrics['bpb']:.3f}"
        )
        print(
            f"🧠 Spike: {metrics['spike_ratio']*100:.0f}% | 🎯 Güven: {metrics['confidence']*100:.0f}% | 🔢 Token: {metrics['n_tokens']}"
        )
        print("-" * 70)


if __name__ == "__main__":
    test_inference()

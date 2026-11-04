"""Mergen-SNN — Chat Model Test Çıkarımı (Stage 5 21M & Stage 4)."""

import argparse
import math
from pathlib import Path
import sys
import time

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

import torch
from tokenizers import Tokenizer

from mergen.models.snn_lm_bpe import BpeSNNLM

# Windows Unicode konsol uyumluluğu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SYSTEM_PROMPT = "Sen Mergen'sin. Bilgisayar bilimi uzmanısın. CS dışı konularda kibarca reddet."
DEFAULT_PROMPTS = [
    "Merhaba",
    "Python'da liste nedir?",
    "Hava nasıl?",
    "def train(",
]


def test_inference(checkpoint_path: Path = None, prompt: str = None, device_str: str = "auto"):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if checkpoint_path is None:
        candidates = [
            Path("checkpoints/stage5_21m_sft.pt"),
            Path("checkpoints/mergen_sft.pt"),
            Path("checkpoints/stage5_21m_final.pt"),
        ]
        for c in candidates:
            if c.exists():
                checkpoint_path = c
                break
        if checkpoint_path is None:
            checkpoint_path = Path("checkpoints/stage5_21m_sft.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    is_stage5 = "stage5" in checkpoint_path.name.lower() or checkpoint.get("vocab_size", 0) >= 16384

    if is_stage5 and Path("data/processed/tokenizer_bpe_v2.json").exists():
        tok_path = "data/processed/tokenizer_bpe_v2.json"
        system_tok = "<system>"
        user_tok = "<user>"
        asst_tok = "<assistant>"
        end_tok = "<endoftext>"
    else:
        tok_path = "data/processed/tokenizer_bpe.json"
        system_tok = "<|system|>"
        user_tok = "<|user|>"
        asst_tok = "<|assistant|>"
        end_tok = "<|end|>"

    tokenizer = Tokenizer.from_file(tok_path)
    end_token_id = tokenizer.token_to_id(end_tok) or tokenizer.token_to_id("<endoftext>") or 2

    vocab_size = checkpoint.get("vocab_size", 16384 if is_stage5 else 8196)
    embed_dim = checkpoint.get("embed_dim", 640 if is_stage5 else 384)
    hidden_size = checkpoint.get("hidden_size", 640 if is_stage5 else 384)
    beta = checkpoint.get("beta", 0.9)
    tie_weights = checkpoint.get("tie_weights", False)

    model = BpeSNNLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        beta=beta,
        tie_weights=tie_weights,
        use_cython=(device.type == "cpu"),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    print("=" * 70)
    print("MERGEN-SNN CHAT TEST ÇIKARIMI")
    print(f"Checkpoint : {checkpoint_path} (Cihaz: {device})")
    print(f"Tokenizer  : {tok_path} (Vocab: {vocab_size})")
    print("=" * 70)

    prompts = [prompt] if prompt else DEFAULT_PROMPTS

    for i, p in enumerate(prompts, 1):
        formatted = f"{system_tok}{SYSTEM_PROMPT}{end_tok}{user_tok}{p}{end_tok}{asst_tok}"
        enc = tokenizer.encode(formatted)

        t0 = time.time()
        output_ids, metrics = model.generate_with_metrics(
            enc,
            max_new_tokens=60,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            end_token_id=end_token_id,
        )
        elapsed_ms = (time.time() - t0) * 1000

        # Sadece asistan yanıtını decode et
        response = tokenizer.decode(output_ids).strip()

        print(f"\n[Örnek {i}]")
        print(f"Kullanıcı Mesajı (Prompt): {p!r}")
        print(f"Asistan Yanıtı (Response):\n{response}")
        print("\nMetrikler:")
        print(
            f"⏱️ Süre: {elapsed_ms:.0f}ms | 📊 Loss: {metrics.get('loss', 0.0):.3f} | PPL: {metrics.get('ppl', 0.0):.1f} | BPB: {metrics.get('bpb', 0.0):.3f}"
        )
        print(
            f"🧠 Spike: {metrics.get('spike_ratio', 0.0)*100:.0f}% | 🎯 Güven: {metrics.get('confidence', 0.0)*100:.0f}% | 🔢 Token: {metrics.get('n_tokens', 0)}"
        )
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Mergen-SNN Chat Çıkarım Testi")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint yolu")
    parser.add_argument("--prompt", type=str, default=None, help="Özel kullanıcı promptu")
    parser.add_argument("--device", type=str, default="auto", help="Cihaz (auto / cuda / cpu)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    test_inference(checkpoint_path=ckpt, prompt=args.prompt, device_str=args.device)


if __name__ == "__main__":
    main()

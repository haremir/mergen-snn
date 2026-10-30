"""Mergen-SNN — Gradio Tabanlı Chat UI.

CS odaklı SNN dil modeli için interaktif arayüz.
Metrikler (süre, loss, PPL, BPB, spike ratio, güven, token sayısı) her mesaj altında görüntülenir.
"""

from pathlib import Path
import sys
import time

import gradio as gr
import torch
from tokenizers import Tokenizer

from mergen.models.snn_lm_bpe import BpeSNNLM

# Windows Unicode konsol uyumluluğu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SYSTEM_PROMPT = "Sen Mergen'sin. Bilgisayar bilimi uzmanısın. CS dışı konularda kibarca reddet."
TOKENIZER_PATH = Path("data/processed/tokenizer_bpe.json")
CHECKPOINT_PATH = Path("checkpoints/mergen_sft.pt")
FALLBACK_CHECKPOINT = Path("checkpoints/bpe_lm_best.pt")

# Tokenizer ve model yükleme
print("Tokenizer yükleniyor...")
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

ckpt_to_load = CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else FALLBACK_CHECKPOINT
print(f"Model yükleniyor: {ckpt_to_load}...")
checkpoint = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)

vocab_size = checkpoint.get("vocab_size", 8196)
embed_dim = checkpoint.get("embed_dim", 384)
hidden_size = checkpoint.get("hidden_size", 384)
beta = checkpoint.get("beta", 0.9)

model = BpeSNNLM(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    hidden_size=hidden_size,
    beta=beta,
    tie_weights=False,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Model hazır! Toplam parametre: {sum(p.numel() for p in model.parameters()):,}")


def generate_response(message: str, history=None, temperature: float = 0.8) -> str:
    """Kullanıcı mesajına SNN modeli ile yanıt üretir ve altına metrik kartını ekler."""
    if not message or not message.strip():
        return "Lütfen bir soru veya komut girin."

    start = time.time()

    # Format: <|system|>{system}<|end|><|user|>{user}<|end|><|assistant|>
    prompt_str = f"<|system|>{SYSTEM_PROMPT}<|end|><|user|>{message.strip()}<|end|><|assistant|>"
    tokens = tokenizer.encode(prompt_str)

    # Model çıkarımı ve metrik hesaplama
    output, metrics = model.generate_with_metrics(
        tokens,
        max_new_tokens=100,
        temperature=temperature,
        top_k=40,
        top_p=0.9,
        end_token_id=8195,
    )
    elapsed = (time.time() - start) * 1000

    response = tokenizer.decode(output).strip()
    if not response:
        response = "(Boş yanıt)"

    # Metrik formatı
    metrics_text = f"""

---
⏱️ Süre: {elapsed:.0f}ms | 📊 Loss: {metrics['loss']:.3f} | PPL: {metrics['ppl']:.1f} | BPB: {metrics['bpb']:.3f}
🧠 Spike: {metrics['spike_ratio']*100:.0f}% | 🎯 Güven: {metrics['confidence']*100:.0f}% | 🔢 Token: {metrics['n_tokens']}"""

    return response + metrics_text


def create_demo():
    demo = gr.ChatInterface(
        fn=generate_response,
        title="Mergen-SNN Chat",
        description="CS odaklı SNN dil modeli. Metrikler her mesaj altında görünür.",
        examples=["Merhaba", "Python'da liste nedir?", "Hava nasıl?", "def train("],
    )
    return demo


if __name__ == "__main__":
    demo = create_demo()
    print("Gradio sunucusu başlatılıyor: http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860)

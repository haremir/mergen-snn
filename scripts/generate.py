"""Mergen-SNN — Stage 2 Metin Üretim Scripti (CharSNNLM).

Gelişmiş örnekleme stratejileri (Sıcaklık, Top-k, Top-p/Nucleus) ve REPL modu.
"""

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional

import torch
import yaml

from mergen.data.char_dataset import decode, encode, load_charset
from mergen.models.snn_lm import CharSNNLM
from mergen.utils.device import device_name, get_device
from mergen.utils.logger import get_logger

# Windows konsolunda Unicode çıktı uyumluluğu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = get_logger("mergen.generate")


def load_config() -> dict:
    """default.yaml ve stage2_char.yaml yapılandırmalarını birleştirir."""
    configs_dir = Path("configs")
    default_cfg_path = configs_dir / "default.yaml"
    stage2_cfg_path = configs_dir / "stage2_char.yaml"

    config: Dict = {}
    if default_cfg_path.exists():
        with open(default_cfg_path, "r", encoding="utf-8") as f:
            config.update(yaml.safe_load(f) or {})

    if stage2_cfg_path.exists():
        with open(stage2_cfg_path, "r", encoding="utf-8") as f:
            stage2_cfg = yaml.safe_load(f) or {}
            for k, v in stage2_cfg.items():
                if isinstance(v, dict) and k in config and isinstance(config[k], dict):
                    config[k].update(v)
                else:
                    config[k] = v

    return config


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.9,
) -> int:
    """Sıcaklık, Top-k ve Top-p (nucleus) filtrelemesi ile logits üzerinden token örnekler."""
    temp = max(float(temperature), 1e-4)
    filtered_logits = logits.clone() / temp

    # 1. Top-k filtreleme
    if top_k > 0:
        k = min(top_k, filtered_logits.size(-1))
        v, _ = torch.topk(filtered_logits, k)
        filtered_logits[filtered_logits < v[-1]] = -float("Inf")

    # 2. Top-p (nucleus) filtreleme
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # p eşiğini aşanları filtrele (ilk elemanı koru)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        filtered_logits[indices_to_remove] = -float("Inf")

    # 3. Softmax ve multinomial örnekleme
    probs = torch.softmax(filtered_logits, dim=-1)
    if torch.isnan(probs).any() or probs.sum() == 0:
        probs = torch.ones_like(probs) / probs.size(-1)

    return torch.multinomial(probs, num_samples=1).item()


def generate(
    model: torch.nn.Module,
    prompt: str,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    device: torch.device,
    max_new_chars: int = 80,
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.9,
    num_steps: int = 32,
    block_size: int = 32,
) -> str:
    """Sliding window ve Top-k/Top-p/Sıcaklık örneklemesi ile prompt üzerinden yeni karakterler üretir."""
    model.eval()
    unk_id = stoi.get("<unk>", 0)

    # 1. Prompt'u encode et
    current_ids = encode(prompt, stoi)

    with torch.no_grad():
        for _ in range(max_new_chars):
            # 2. Blok boyutuna göre kayan pencere (sliding window) ve pad
            if len(current_ids) < block_size:
                pad_len = block_size - len(current_ids)
                window_ids = [unk_id] * pad_len + current_ids
            else:
                window_ids = current_ids[-block_size:]

            inp = torch.tensor([window_ids], dtype=torch.long, device=device)

            # 3. Model forward
            logits = model(inp, num_steps=block_size)  # (1, block_size, vocab_size)

            # 4. Son timestep logits'i al ve örnekle
            last_logits = logits[0, -1, :]
            next_id = sample_token(
                logits=last_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            current_ids.append(next_id)

    # 5. Dönen: decode edilmiş tam metin
    return decode(current_ids, itos)


def run_repl(
    model: torch.nn.Module,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    device: torch.device,
    block_size: int,
    top_k: int,
    top_p: float,
):
    """Kullanıcı ile etkileşimli REPL metin üretim modu."""
    print("\n" + "=" * 60)
    print("MERGEN-SNN REPL MODU (Çıkmak için 'q' veya 'exit' yazın)")
    print("=" * 60)
    while True:
        try:
            prompt = input("\nPrompt > ")
            if not prompt or prompt.strip().lower() in ("q", "exit"):
                print("REPL modundan çıkılıyor...")
                break
            temp_str = input("Sıcaklık (varsayılan: 0.8) > ").strip()
            temp = float(temp_str) if temp_str else 0.8
            length_str = input("Üretilecek karakter sayısı (varsayılan: 100) > ").strip()
            max_chars = int(length_str) if length_str else 100

            output = generate(
                model=model,
                prompt=prompt,
                stoi=stoi,
                itos=itos,
                device=device,
                max_new_chars=max_chars,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                num_steps=block_size,
                block_size=block_size,
            )
            print("-" * 50)
            print(output)
            print("-" * 50)
        except (KeyboardInterrupt, EOFError):
            print("\nREPL modundan çıkılıyor...")
            break


def main():
    parser = argparse.ArgumentParser(description="Mergen-SNN Metin Üretimi")
    parser.add_argument("--repl", action="store_true", help="Etkileşimli REPL modunu başlat")
    parser.add_argument("--prompt", type=str, default=None, help="Özel metin istemi")
    parser.add_argument("--temperature", type=float, default=None, help="Örnekleme sıcaklığı")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k parametresi")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p parametresi")
    parser.add_argument("--max_chars", type=int, default=None, help="Üretilecek yeni karakter sayısı")
    args = parser.parse_args()

    cfg = load_config()

    prefer_device = cfg.get("device", {}).get("prefer", "cpu")
    device = get_device(prefer=prefer_device)
    dev_name = device_name(device)

    data_dir = Path(cfg.get("paths", {}).get("data_processed", "data/processed"))
    checkpoints_dir = Path(cfg.get("paths", {}).get("checkpoints", "checkpoints"))

    block_size = int(cfg.get("data", {}).get("block_size", 32))
    embed_dim = int(cfg.get("model", {}).get("embed_dim", 128))
    hidden_size = int(cfg.get("model", {}).get("hidden_size", 256))
    beta = float(cfg.get("model", {}).get("beta", 0.9))

    max_new_chars = args.max_chars or int(cfg.get("generation", {}).get("max_new_chars", 80))
    top_k = args.top_k or int(cfg.get("generation", {}).get("top_k", 40))
    top_p = args.top_p or float(cfg.get("generation", {}).get("top_p", 0.9))
    base_temp = args.temperature or float(cfg.get("generation", {}).get("temperature", 1.0))

    # Charset yükle
    charset_path = data_dir / "charset.json"
    stoi, itos = load_charset(charset_path)
    vocab_size = len(stoi)

    # Checkpoint bul ve yükle: lm_best.pt > lm_final.pt > lm_epoch*.pt
    best_ckpt = checkpoints_dir / "lm_best.pt"
    final_ckpt = checkpoints_dir / "lm_final.pt"

    if best_ckpt.exists():
        ckpt_path = best_ckpt
    elif final_ckpt.exists():
        ckpt_path = final_ckpt
    else:
        ckpts = sorted(checkpoints_dir.glob("lm_epoch*.pt"))
        if not ckpts:
            logger.error(f"Eğitilmiş model checkpoint'i bulunamadı: {checkpoints_dir}")
            return
        ckpt_path = ckpts[-1]

    logger.info(f"Checkpoint yükleniyor: {ckpt_path} (Cihaz: {dev_name})")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = CharSNNLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        beta=beta,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.repl:
        run_repl(
            model=model,
            stoi=stoi,
            itos=itos,
            device=device,
            block_size=block_size,
            top_k=top_k,
            top_p=top_p,
        )
        return

    # Çoklu sıcaklık ve prompt testi
    temperatures = [0.7, 0.8, 0.9, 1.0, 1.2]
    prompts = [
        "Python'da liste",
        "Algoritma nedir",
        "Bilgisayar mimarisi",
    ]

    print("\n" + "=" * 70)
    print("MERGEN-SNN — STAGE 2 METİN ÜRETİMİ (SNN Dil Modeli)")
    print(f"Model: {ckpt_path.name} | Top-k: {top_k} | Top-p: {top_p} | max_new_chars: {max_new_chars}")
    print("=" * 70)

    for p_idx, p in enumerate(prompts, 1):
        print(f"\n{'#' * 70}")
        print(f"PROMPT {p_idx}: \"{p}\"")
        print(f"{'#' * 70}")
        for temp in temperatures:
            print(f"\n--- Sıcaklık (Temperature): {temp:.1f} ---")
            generated_text = generate(
                model=model,
                prompt=p,
                stoi=stoi,
                itos=itos,
                device=device,
                max_new_chars=max_new_chars,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                num_steps=block_size,
                block_size=block_size,
            )
            print(f"Çıktı:\n{generated_text}")
        print("-" * 70)


if __name__ == "__main__":
    main()

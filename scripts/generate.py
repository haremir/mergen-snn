"""Mergen-SNN — Stage 2 Metin Üretim Scripti (CharSNNLM)."""

import os
from pathlib import Path
import sys
from typing import Dict, List

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


def generate(
    model: torch.nn.Module,
    prompt: str,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    device: torch.device,
    max_new_chars: int = 80,
    temperature: float = 1.0,
    num_steps: int = 32,
    block_size: int = 32,
) -> str:
    """Sliding window ve sıcaklık örneklemesi ile prompt üzerinden yeni karakterler üretir."""
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

            # 4. Son timestep logits'i al ve sıcaklık uygula
            last_logits = logits[0, -1, :]
            temp = max(float(temperature), 1e-4)
            scaled_logits = last_logits / temp

            # 5. Olasılık dağılımı ve multinomial örnekleme
            probs = torch.softmax(scaled_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            current_ids.append(next_id)

    # 6. Dönen: decode edilmiş tam metin
    return decode(current_ids, itos)


def main():
    cfg = load_config()

    prefer_device = cfg.get("device", {}).get("prefer", "auto")
    device = get_device(prefer=prefer_device)
    dev_name = device_name(device)

    data_dir = Path(cfg.get("paths", {}).get("data_processed", "data/processed"))
    checkpoints_dir = Path(cfg.get("paths", {}).get("checkpoints", "checkpoints"))

    block_size = int(cfg.get("data", {}).get("block_size", 32))
    embed_dim = int(cfg.get("model", {}).get("embed_dim", 128))
    hidden_size = int(cfg.get("model", {}).get("hidden_size", 256))
    beta = float(cfg.get("model", {}).get("beta", 0.9))

    max_new_chars = int(cfg.get("generation", {}).get("max_new_chars", 80))
    temperature = float(cfg.get("generation", {}).get("temperature", 1.0))

    # Charset yükle
    charset_path = data_dir / "charset.json"
    stoi, itos = load_charset(charset_path)
    vocab_size = len(stoi)

    # Checkpoint bul ve yükle
    best_ckpt = checkpoints_dir / "lm_best.pt"
    epoch1_ckpt = checkpoints_dir / "lm_epoch1.pt"

    if best_ckpt.exists():
        ckpt_path = best_ckpt
    elif epoch1_ckpt.exists():
        ckpt_path = epoch1_ckpt
    else:
        # Herhangi bir lm_epoch*.pt ara
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

    prompts = [
        "Python'da liste",
        "Algoritma nedir",
        "Bilgisayar mimarisi",
    ]

    print("\n" + "=" * 60)
    print("MERGEN-SNN — STAGE 2 METİN ÜRETİMİ (SNN Dil Modeli)")
    print(f"Model: {ckpt_path.name} | Sıcaklık: {temperature} | max_new_chars: {max_new_chars}")
    print("=" * 60)

    for i, p in enumerate(prompts, 1):
        print(f"\n[Örnek {i}]")
        print(f"Girdi (Prompt) : {p}")
        generated_text = generate(
            model=model,
            prompt=p,
            stoi=stoi,
            itos=itos,
            device=device,
            max_new_chars=max_new_chars,
            temperature=temperature,
            num_steps=block_size,
            block_size=block_size,
        )
        print(f"Çıktı (Üretim) : {generated_text}")
        print("-" * 60)


if __name__ == "__main__":
    main()

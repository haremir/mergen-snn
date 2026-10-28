"""Mergen-SNN — Metin Üretim Scripti (Stage 2 Char & Stage 3 BPE).

Gelişmiş örnekleme (Sıcaklık, Top-k, Top-p/Nucleus) ve hem Char hem BPE desteği.
"""

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union

import torch
import yaml

from mergen.data.char_dataset import decode as char_decode, encode as char_encode, load_charset
from mergen.models.snn_lm import CharSNNLM
from mergen.models.snn_lm_bpe import BpeSNNLM
from mergen.utils.device import device_name, get_device
from mergen.utils.logger import get_logger

# Windows konsolunda Unicode çıktı uyumluluğu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = get_logger("mergen.generate")


def load_config(config_file: Optional[str] = None) -> dict:
    """default.yaml ve ilgili stage config dosyasını birleştirir."""
    configs_dir = Path("configs")
    default_cfg_path = configs_dir / "default.yaml"

    if config_file:
        stage_cfg_path = Path(config_file)
    else:
        stage3_path = configs_dir / "stage3_bpe.yaml"
        stage2_path = configs_dir / "stage2_char.yaml"
        stage_cfg_path = stage3_path if stage3_path.exists() else stage2_path

    config: Dict = {}
    if default_cfg_path.exists():
        with open(default_cfg_path, "r", encoding="utf-8") as f:
            config.update(yaml.safe_load(f) or {})

    if stage_cfg_path.exists():
        with open(stage_cfg_path, "r", encoding="utf-8") as f:
            stage_cfg = yaml.safe_load(f) or {}
            for k, v in stage_cfg.items():
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


def generate_bpe(
    model: torch.nn.Module,
    prompt: str,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    block_size: int = 128,
) -> str:
    """Byte-Level BPE tokenizer ile sliding window metin üretimi."""
    model.eval()
    pad_id = tokenizer.token_to_id("<pad>") or 0

    encoded = tokenizer.encode(prompt)
    current_ids = list(encoded.ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(current_ids) < block_size:
                pad_len = block_size - len(current_ids)
                window_ids = [pad_id] * pad_len + current_ids
            else:
                window_ids = current_ids[-block_size:]

            inp = torch.tensor([window_ids], dtype=torch.long, device=device)
            logits = model(inp, num_steps=block_size)

            last_logits = logits[0, -1, :]
            next_id = sample_token(
                logits=last_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            current_ids.append(next_id)

    return tokenizer.decode(current_ids)


def generate_char(
    model: torch.nn.Module,
    prompt: str,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    device: torch.device,
    max_new_chars: int = 80,
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.9,
    block_size: int = 32,
) -> str:
    """Karakter seviyesinde sliding window metin üretimi."""
    model.eval()
    unk_id = stoi.get("<unk>", 0)
    current_ids = char_encode(prompt, stoi)

    with torch.no_grad():
        for _ in range(max_new_chars):
            if len(current_ids) < block_size:
                pad_len = block_size - len(current_ids)
                window_ids = [unk_id] * pad_len + current_ids
            else:
                window_ids = current_ids[-block_size:]

            inp = torch.tensor([window_ids], dtype=torch.long, device=device)
            logits = model(inp, num_steps=block_size)

            last_logits = logits[0, -1, :]
            next_id = sample_token(
                logits=last_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            current_ids.append(next_id)

    return char_decode(current_ids, itos)


def main():
    parser = argparse.ArgumentParser(description="Mergen-SNN Metin Üretimi")
    parser.add_argument("--config", type=str, default=None, help="YAML config dosyası")
    parser.add_argument("--checkpoint", type=str, default=None, help="Özel checkpoint yolu")
    parser.add_argument("--temperature", type=float, default=None, help="Örnekleme sıcaklığı")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p")
    parser.add_argument("--max_tokens", type=int, default=None, help="Üretilecek token sayısı")
    args = parser.parse_args()

    cfg = load_config(args.config)

    prefer_device = cfg.get("device", {}).get("prefer", "cpu")
    device = get_device(prefer=prefer_device)
    dev_name = device_name(device)

    data_dir = Path(cfg.get("paths", {}).get("data_processed", "data/processed"))
    checkpoints_dir = Path(cfg.get("paths", {}).get("checkpoints", "checkpoints"))

    # Checkpoint seçimi: bpe_lm_best > bpe_lm_epoch* > lm_best > lm_epoch*
    ckpt_path = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        candidates = [
            checkpoints_dir / "bpe_lm_best.pt",
            checkpoints_dir / "bpe_lm_epoch1.pt",
            checkpoints_dir / "bpe_lm_final.pt",
            checkpoints_dir / "lm_best.pt",
            checkpoints_dir / "lm_epoch2.pt",
            checkpoints_dir / "lm_epoch1.pt",
        ]
        for c in candidates:
            if c.exists():
                ckpt_path = c
                break

    if not ckpt_path or not ckpt_path.exists():
        logger.error(f"Hiçbir checkpoint bulunamadı: {checkpoints_dir}")
        return

    logger.info(f"Checkpoint yükleniyor: {ckpt_path.name} (Cihaz: {dev_name})")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    is_bpe = checkpoint.get("is_bpe", False) or "bpe" in ckpt_path.name.lower()
    vocab_size = checkpoint.get("vocab_size", 8192 if is_bpe else 460)
    embed_dim = checkpoint.get("embed_dim", 512 if is_bpe else 128)
    hidden_size = checkpoint.get("hidden_size", 1024 if is_bpe else 256)
    beta = checkpoint.get("beta", 0.9)

    block_size = int(cfg.get("data", {}).get("block_size", 128 if is_bpe else 32))
    temperature = args.temperature or float(cfg.get("generation", {}).get("temperature", 0.8))
    top_k = args.top_k or int(cfg.get("generation", {}).get("top_k", 40))
    top_p = args.top_p or float(cfg.get("generation", {}).get("top_p", 0.9))
    max_new_tokens = args.max_tokens or int(cfg.get("generation", {}).get("max_new_tokens", 60))

    if is_bpe:
        from tokenizers import Tokenizer
        tokenizer_path = data_dir / "tokenizer_bpe.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        model = BpeSNNLM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            beta=beta,
        ).to(device)
    else:
        charset_path = data_dir / "charset.json"
        stoi, itos = load_charset(charset_path)
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
        "def train(",
    ]

    print("\n" + "=" * 70)
    print(f"MERGEN-SNN — {'STAGE 3 BPE (10M)' if is_bpe else 'STAGE 2 CHAR'} METİN ÜRETİMİ")
    print(f"Model: {ckpt_path.name} | Sıcaklık: {temperature} | Top-k: {top_k} | Top-p: {top_p}")
    print("=" * 70)

    for i, p in enumerate(prompts, 1):
        print(f"\n[Örnek {i}]")
        print(f"Girdi (Prompt) : {p!r}")
        if is_bpe:
            out = generate_bpe(
                model=model,
                prompt=p,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                block_size=block_size,
            )
        else:
            out = generate_char(
                model=model,
                prompt=p,
                stoi=stoi,
                itos=itos,
                device=device,
                max_new_chars=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                block_size=block_size,
            )
        print(f"Çıktı (Üretim) :\n{out}")
        print("-" * 70)


if __name__ == "__main__":
    main()

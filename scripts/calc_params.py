"""Calculate and display exact parameter breakdown for Mergen-SNN Stage 6 (50M)."""

import argparse
from pathlib import Path
import sys

# Add src to pythonpath
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

import yaml
from mergen.models.snn_lm_bpe import BpeSNNLM


def calculate_parameters(config_path: str = "configs/stage6_50m_16k.yaml"):
    path = Path(config_path)
    if not path.exists():
        print(f"Hata: Yapılandırma dosyası bulunamadı: {config_path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mcfg = cfg.get("model", {})
    vocab_size = mcfg.get("vocab_size", 16384)
    embed_dim = mcfg.get("embed_dim", 1400)
    hidden_size = mcfg.get("hidden_size", 1400)
    beta = mcfg.get("beta", 0.97)
    tie_weights = mcfg.get("tie_weights", False)
    use_cython = mcfg.get("use_cython", False)

    print("==================================================================")
    print(f"Mergen-SNN Model Parametre Hesaplayıcı — Yapılandırma: {config_path}")
    print("==================================================================")
    print(f"Vocab Size   : {vocab_size:,}")
    print(f"Embed Dim    : {embed_dim}")
    print(f"Hidden Size  : {hidden_size}")
    print(f"Beta         : {beta}")
    print(f"Tie Weights  : {tie_weights}")
    print("------------------------------------------------------------------")

    model = BpeSNNLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        beta=beta,
        tie_weights=tie_weights,
        use_cython=use_cython,
    )

    total_params = 0
    print(f"{'Katman Adı':<28} {'Şekil':<20} {'Parametre Sayısı':>16}")
    print("-" * 66)
    for name, p in model.named_parameters():
        num = p.numel()
        total_params += num
        shape_str = str(list(p.shape))
        print(f"{name:<28} {shape_str:<20} {num:>16,}")

    print("=" * 66)
    print(f"{'TOPLAM PARAMETRE SAYISI':<49} {total_params:>16,}")
    print(f"{'YAKLAŞIK BOYUT (Milyon)':<49} {total_params / 1e6:>16.2f}M")
    print("==================================================================")
    return total_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage6_50m_16k.yaml")
    args = parser.parse_args()
    calculate_parameters(args.config)

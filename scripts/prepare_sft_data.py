"""Mergen-SNN — SFT Veri Seti Hazırlama ve Formatlama Scripti.

Her örneği şu formata çevirir:
<|system|>{system}<|end|><|user|>{user}<|end|><|assistant|>{assistant}<|end|>

Çıktılar:
- data/sft/sft_train.bin: uint16 dizi (num_samples, max_seq_len)
- data/sft/sft_mask.bin: uint8 kayıp maskesi (num_samples, max_seq_len - 1)
- data/sft/sft_meta.json: Özet istatistikler
"""

import json
from pathlib import Path
import sys

import numpy as np
from tokenizers import Tokenizer

# Windows Unicode uyumluluğu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def prepare_sft_data(
    jsonl_path: Path = Path("data/sft/train.jsonl"),
    tokenizer_path: Path = Path("data/processed/tokenizer_bpe.json"),
    output_dir: Path = Path("data/sft"),
    max_seq_len: int = 256,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer bulunamadı: {tokenizer_path}")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"SFT JSONL veri seti bulunamadı: {jsonl_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Tokenizer özel token kimlikleri
    pad_id = tokenizer.token_to_id("<pad>") or 1
    system_tok = "<|system|>"
    user_tok = "<|user|>"
    asst_tok = "<|assistant|>"
    end_tok = "<|end|>"

    for tok_name in [system_tok, user_tok, asst_tok, end_tok]:
        if tokenizer.token_to_id(tok_name) is None:
            raise ValueError(f"Özel token bulunamadı: {tok_name}. Lütfen tokenizer'ı güncelleyin.")

    samples_input_ids = []
    samples_loss_masks = []

    total_tokens_all = 0
    total_assistant_tokens = 0
    truncated_count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    print(f"Toplam {len(lines)} örnek işleniyor...")

    for i, item in enumerate(lines):
        system_text = item.get("system", "").strip()
        user_text = item.get("user", "").strip()
        asst_text = item.get("assistant", "").strip()

        prefix_str = f"{system_tok}{system_text}{end_tok}{user_tok}{user_text}{end_tok}{asst_tok}"
        asst_str = f"{asst_text}{end_tok}"
        full_str = f"{prefix_str}{asst_str}"

        p_ids = tokenizer.encode(prefix_str).ids
        f_ids = tokenizer.encode(full_str).ids

        total_tokens_all += len(f_ids)

        if len(f_ids) > max_seq_len:
            truncated_count += 1
            f_ids = f_ids[:max_seq_len]

        # Padding (sağdan dolgu)
        pad_len = max_seq_len - len(f_ids)
        padded_ids = f_ids + [pad_id] * pad_len

        # Maske hesaplama: Causal LM'de x = padded_ids[:-1], y = padded_ids[1:]
        # y[t] = padded_ids[t + 1] hedefinin asistan yanıtına ait olduğu adımlar
        # Asistan aralığı: p_ids uzunluğundan f_ids sonuna kadar (t + 1 in [len(p_ids), len(f_ids)])
        mask = [0] * (max_seq_len - 1)
        start_idx = max(0, len(p_ids) - 1)
        end_idx = min(len(f_ids) - 1, max_seq_len - 1)

        for t in range(start_idx, end_idx):
            mask[t] = 1
            total_assistant_tokens += 1

        samples_input_ids.append(padded_ids)
        samples_loss_masks.append(mask)

    arr_input = np.array(samples_input_ids, dtype=np.uint16)
    arr_mask = np.array(samples_loss_masks, dtype=np.uint8)

    train_bin_path = output_dir / "sft_train.bin"
    mask_bin_path = output_dir / "sft_mask.bin"
    meta_json_path = output_dir / "sft_meta.json"

    arr_input.tofile(train_bin_path)
    arr_mask.tofile(mask_bin_path)

    meta = {
        "num_samples": len(lines),
        "max_seq_len": max_seq_len,
        "input_bin": str(train_bin_path.name),
        "mask_bin": str(mask_bin_path.name),
        "total_tokens_raw": total_tokens_all,
        "total_assistant_supervised_tokens": total_assistant_tokens,
        "average_tokens_per_sample": round(total_tokens_all / max(len(lines), 1), 2),
        "average_assistant_tokens_per_sample": round(total_assistant_tokens / max(len(lines), 1), 2),
        "truncated_samples": truncated_count,
        "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
    }

    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("SFT VERİ SETİ HAZIRLAMA TAMAMLANDI")
    print("=" * 60)
    print(f"Toplam Örnek Sayısı          : {meta['num_samples']}")
    print(f"Maksimum Dizi Uzunluğu (Seq) : {meta['max_seq_len']}")
    print(f"Toplam Ham Token             : {meta['total_tokens_raw']}")
    print(f"Süpervize Edilen Token (Asst): {meta['total_assistant_supervised_tokens']}")
    print(f"Ortalama Token / Örnek       : {meta['average_tokens_per_sample']}")
    print(f"Kırpılan (Truncated) Örnek   : {meta['truncated_samples']}")
    print(f"Kaydedilen Dosyalar          : {train_bin_path}, {mask_bin_path}, {meta_json_path}")
    print("=" * 60)


if __name__ == "__main__":
    prepare_sft_data()

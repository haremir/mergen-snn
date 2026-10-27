"""Mergen-SNN — CharDataset ve Charset Testleri.

Karakter kodlama/kod çözme roundtrip doğrulaması, Dataset tensör boyutları ve kaydırma mantığı.
"""

from pathlib import Path
import pytest
import torch

from mergen.data.char_dataset import CharDataset, decode, encode, load_charset

CHARSET_PATH = Path("data/processed/charset.json")
TRAIN_BIN_PATH = Path("data/processed/train.bin")


def test_charset_roundtrip():
    """Küçük metin encode -> decode eşit olsun."""
    assert CHARSET_PATH.exists(), f"charset.json bulunamadı: {CHARSET_PATH}"
    stoi, itos = load_charset(CHARSET_PATH)

    sample_text = "def forward(x, num_steps=25):\n    return x * 2"
    encoded = encode(sample_text, stoi)
    decoded = decode(encoded, itos)
    assert decoded == sample_text


def test_dataset_shapes():
    """x.shape == (32,), y.shape == (32,), torch.equal(y[:-1], x[1:]) kaydırma kontrolü."""
    assert TRAIN_BIN_PATH.exists(), f"train.bin bulunamadı: {TRAIN_BIN_PATH}"
    dataset = CharDataset(TRAIN_BIN_PATH, block_size=32)
    assert len(dataset) > 0

    x, y = dataset[0]
    assert x.shape == (32,)
    assert y.shape == (32,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    # Karakter dil modeli hedef kaydırma kuralı: y[i] == x[i+1]
    assert torch.equal(y[:-1], x[1:])


def test_turkish_chars():
    """Türkçe karakterler (çÇğĞıİöÖşŞüÜ) ve <unk> tokeni her zaman charset'te bulunmalıdır."""
    assert CHARSET_PATH.exists(), f"charset.json bulunamadı: {CHARSET_PATH}"
    stoi, _ = load_charset(CHARSET_PATH)

    required_turkish = "çÇğĞıİöÖşŞüÜ"
    missing = [ch for ch in required_turkish if ch not in stoi]
    assert not missing, f"Türkçe karakterler eksik: {missing}"
    assert "<unk>" in stoi, "<unk> token yok"

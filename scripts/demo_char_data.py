"""Mergen-SNN — Karakter Veri Kümesi Demo Betiği.

CharDataset ve charset.json'u test eder, bir mini-batch görselleştirir,
Türkçe roundtrip kontrolü yapar ve karakter setini denetler.
"""

from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

from mergen.data.char_dataset import CharDataset, decode, encode, load_charset

# Windows konsolunda Unicode karakterlerin düzgün basılması için
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    charset_path = Path("data/processed/charset.json")
    train_bin_path = Path("data/processed/train.bin")

    if not charset_path.exists() or not train_bin_path.exists():
        print("HATA: data/processed/ altında charset.json veya train.bin bulunamadı!")
        print("Lütfen önce 'uv run python scripts/prepare_data.py' çalıştırın.")
        return

    # 1. Charset yükle
    stoi, itos = load_charset(charset_path)
    print(f"Charset başarıyla yüklendi. Vocab size: {len(stoi)}")

    # 2. Veri kümesini başlat
    dataset = CharDataset(train_bin_path, block_size=32)
    print(f"CharDataset yüklendi. Toplam örnek sayısı: {len(dataset):,}")

    # 3. DataLoader oluştur
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 4. 1 batch al ve konsola yaz
    batch_x, batch_y = next(iter(dataloader))
    print("\n" + "=" * 70)
    print("DEMO BATCH (batch_size=4, block_size=32):")
    print("=" * 70)
    for i in range(len(batch_x)):
        x_str = decode(batch_x[i], itos)
        y_str = decode(batch_y[i], itos)
        print(f"Örnek {i + 1}:")
        print(f"  Girdi: {x_str!r}")
        print(f"  Hedef: {y_str!r}")
    print("=" * 70 + "\n")

    # 5. Roundtrip Testi
    test_str = "Python'da liste köşeli parantezlerle tanımlanır."
    encoded = encode(test_str, stoi)
    decoded = decode(encoded, itos)

    print(f"Roundtrip Test Metni: {test_str!r}")
    print(f"Encode -> Decode   : {decoded!r}")
    if test_str == decoded:
        print("Roundtrip Testi: BAŞARILI (Metin birebir korundu)")
    else:
        print("Roundtrip Testi: UYARI / FARK (Bazı karakterler charset'te eksik)")

    # 6. Türkçe Karakter Kontrolü (çÇğĞıİöÖşŞüÜ)
    tr_chars = "çÇğĞıİöÖşŞüÜ"
    missing_tr = [c for c in tr_chars if c not in stoi]
    present_tr = [c for c in tr_chars if c in stoi]

    if not missing_tr:
        print(f"Türkçe karakterlerin tamamı ({len(present_tr)}/12) charset'te mevcut ✓ (çÇğĞıİöÖşŞüÜ)")
    else:
        raise ValueError(f"HATA: Türkçe karakterler charset'te eksik! Eksik: {missing_tr}")


if __name__ == "__main__":
    main()

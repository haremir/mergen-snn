"""Mergen-SNN — Stage 3: Byte-Level BPE Tokenizer Görsel ve Mantıksal Doğrulama."""

from pathlib import Path
import sys

from tokenizers import Tokenizer

# Windows konsolunda Unicode çıktı uyumluluğu
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TOKENIZER_PATH = Path("data/processed/tokenizer_bpe.json")


def main():
    if not TOKENIZER_PATH.exists():
        print(f"HATA: Tokenizer dosyası bulunamadı: {TOKENIZER_PATH}")
        print("Lütfen önce 'scripts/train_tokenizer.py' çalıştırın.")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()

    print("=" * 70)
    print("MERGEN-SNN — STAGE 3: BYTE-LEVEL BPE TOKENIZER DEMO & TEST")
    print(f"Yüklenen Tokenizer: {TOKENIZER_PATH} (Vocab Size: {vocab_size})")
    print("=" * 70)

    test_sentences = [
        ("Türkçe Doğal Dil", "Python'da liste köşeli parantezlerle tanımlanır."),
        ("Python Kodu", "def train(model, loader): return model"),
        ("C Programlama Kodu", "#include <stdio.h> int main() { return 0; }"),
    ]

    all_passed = True

    for i, (category, sentence) in enumerate(test_sentences, 1):
        print(f"\n[Test {i}: {category}]")
        print(f"Orijinal Metin   : {sentence}")

        encoding = tokenizer.encode(sentence)
        tokens = encoding.tokens
        token_ids = encoding.ids
        token_count = len(token_ids)
        char_count = len(sentence)
        comp_ratio = char_count / max(token_count, 1)

        print(f"Karakter Sayısı  : {char_count}")
        print(f"Token Sayısı     : {token_count} (Sıkıştırma: {comp_ratio:.2f} kar/token)")
        print(f"Token Dizisi     : {tokens}")
        print(f"Token ID'leri    : {token_ids}")

        decoded = tokenizer.decode(token_ids)
        print(f"Geri Çözülen     : {decoded}")

        is_exact = (decoded == sentence)
        status_str = "BAŞARILI ✓ (Birebir Eşleşti)" if is_exact else "BAŞARISIZ ✗ (Farklılık Var)"
        print(f"Doğrulama Sonucu : {status_str}")
        print("-" * 70)

        if not is_exact:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("TÜM BPE TESTLERİ BAŞARIYLA TAMAMLANDI! (Roundtrip doğrulandı)")
    else:
        print("DİKKAT: Bazı testlerde çözme farklılığı tespit edildi!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

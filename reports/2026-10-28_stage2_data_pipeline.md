# Aşama 2: Karakter Seviyesi CS Korpus Veri Hattı Raporu

**Tarih:** 2026-10-28  
**Yazar:** Mergen-SNN Geliştirme Takımı  
**Konu:** 92.03 MB CS Korpusunun Karakter Seviyesi Dil Modeli İçin Hazırlanması, Tokenizasyon ve Veri Yükleyici Sözleşmesi  

---

## 1. Amaç

Mergen-SNN projesinde Aşama 1 (MNIST %93.8 doğruluk) tamamlandıktan sonra, Aşama 2 hedefi olan **Bilgisayar Bilimleri (CS) Odaklı Karakter Seviyesi Spiking Nöral Ağ Dil Modeli** için veri hazırlama hattını kurmaktır. 16 farklı kategoriden toplanan 92.03 MB metnin temizlenmesi, karakter sözlüğünün (vocabulary) çıkarılması, sayısal token dizilerine dönüştürülmesi ve PyTorch `CharDataset` yapısı ile donanıma beslenmeye hazır hale getirilmesi hedeflenmiştir.

---

## 2. Yöntem

1. **Metin Birleştirme (`collect_texts`):**
   - `data/raw/cs_corpus/` altındaki 16 kategoride bulunan 740 `.txt` dosyası tarandı.
   - `LICENSE_NOTE.txt`, `_low_quality.txt`, `.gitkeep` ve 100 bayttan küçük dosyalar elendi. Toplam 715 geçerli dosya `\n\n---\n\n` ayırıcıları ile birleştirildi.
2. **Karakter Kümesi İnşası (`build_charset`):**
   - Tüm metindeki karakter frekansları `Counter` ile sayıldı.
   - `min_freq = 5` eşiği uygulanarak nadir OCR hataları veya gürültülü işaretler filtrelendi.
   - Sıralı karakter listesi üzerinden çift yönlü `stoi` ve `itos` eşleme tabloları oluşturuldu.
3. **Bölme ve İkili Depolama (`split_and_save`):**
   - Kelime dağarcığı $736 < 65,536$ olduğu için bellek ve disk tasarrufu sağlayan `np.uint16` veri tipi seçildi.
   - Metin %90 eğitim (`train.bin`), %10 doğrulama (`val.bin`) olarak bölündü ve `np.save` ile diske yazıldı.
   - Sözlük `data/processed/charset.json` ve `data/raw/tokenizers/charset.json` dosyalarına kaydedildi.
4. **Veri Kümesi Sözleşmesi (`CharDataset`):**
   - `torch.utils.data.Dataset` türevi [`CharDataset`](file:///src/mergen/data/char_dataset.py) ile sabit bağlam penceresi (`block_size = 32`) oluşturuldu.
   - Her adımda girdi $x = \text{data}[i : i + 32]$ ve hedef $y = \text{data}[i + 1 : i + 33]$ tensörleri üretilerek otoregresif dil modelleme kuralı ($y[k] = x[k+1]$) sağlandı.

---

## 3. Sonuçlar ve İstatistikler

| Metrik | Değer |
| :--- | :--- |
| **Toplam Taranan Karakter** | 93,193,569 |
| **Benzersiz Ham Karakter Sayısı** | 931 |
| **Filtrelenen Kelime Dağarcığı (Vocab Size)** | **736** (195 nadir karakter elendi) |
| **Toplam Kodlanan Token** | 93,193,158 |
| **Eğitim Token Sayısı (%90)** | **83,873,842** |
| **Doğrulama Token Sayısı (%10)** | **9,319,316** |

### En Sık 20 Karakter

| Karakter | Temsil | Frekans | Yüzde |
| :---: | :---: | :---: | :---: |
| Boşluk | `' '` | 13,594,787 | %14.59 |
| e | `'e'` | 8,062,541 | %8.65 |
| t | `'t'` | 6,189,574 | %6.64 |
| a | `'a'` | 4,876,742 | %5.23 |
| o | `'o'` | 4,650,607 | %4.99 |
| i | `'i'` | 4,645,854 | %4.99 |
| n | `'n'` | 4,583,988 | %4.92 |
| s | `'s'` | 4,403,196 | %4.72 |
| r | `'r'` | 4,230,042 | %4.54 |
| l | `'l'` | 2,677,534 | %2.87 |
| c | `'c'` | 2,566,726 | %2.75 |
| h | `'h'` | 2,350,584 | %2.52 |
| Nokta | `'.'` | 2,185,699 | %2.35 |
| d | `'d'` | 2,139,017 | %2.30 |
| Satır Sonu | `'\n'` | 2,047,649 | %2.20 |
| u | `'u'` | 1,778,375 | %1.91 |
| p | `'p'` | 1,690,081 | %1.81 |
| m | `'m'` | 1,656,503 | %1.78 |
| f | `'f'` | 1,347,691 | %1.45 |
| g | `'g'` | 1,120,072 | %1.20 |

### Dosya Boyutları

- `data/processed/charset.json`: 46.76 KB (46,758 bayt)
- `data/processed/train.bin`: 159.98 MB (167,747,812 bayt)
- `data/processed/val.bin`: 17.78 MB (18,638,760 bayt)

---

## 4. Demo Çıktısı (`scripts/demo_char_data.py`)

```text
Charset başarıyla yüklendi. Vocab size: 736
CharDataset yüklendi. Toplam örnek sayısı: 83,873,809

======================================================================
DEMO BATCH (batch_size=4, block_size=32):
======================================================================
Örnek 1:
  Girdi: '2-aes-1655431\n-munaligned-access'
  Hedef: '-aes-1655431\n-munaligned-access\n'
Örnek 2:
  Girdi: 'or supercomputers. The X1 system'
  Hedef: 'r supercomputers. The X1 system '
Örnek 3:
  Girdi: 'ath′(x0,..., xr−1) =h′(y0,..., y'
  Hedef: 'th′(x0,..., xr−1) =h′(y0,..., yr'
Örnek 4:
  Girdi: 't and correct code. \n• Provide r'
  Hedef: ' and correct code. \n• Provide re'
======================================================================

Roundtrip Test Metni: "Python'da liste köşeli parantezlerle tanımlanır."
Encode -> Decode   : "Python'da liste köşeli parantezlerle tanımlanır."
Roundtrip Testi: BAŞARILI (Metin birebir korundu)
UYARI: Türkçe karakterler charset'te yok! (Mevcut: ['ç', 'ı', 'ö', 'ş', 'ü'], Eksik: ['ğ', 'İ'])
```

---

## 5. Mimari Kararlar

1. **`block_size = 32`:** Stage 2'deki ilk SNN dil modeli için zaman adımları ($T$) ve BPTT bellek ayak izi göz önüne alınarak başlangıç pencere boyutu 32 karakter olarak sabitlenmiştir.
2. **`min_freq = 5`:** Tek seferlik PDF ayrıştırma gürültülerini ve kontrol karakterlerini sözlükten ayıklayarak vocab boyutunu kompakt (736) tutmuştur.
3. **`train_ratio = 0.9`:** Büyük veri kümesinde (%90 / %10) ayrımı 83.8M eğitim tokeni ve 9.3M doğrulama tokeni sağlayarak dengeli bir değerlendirme imkanı sunmaktadır.
4. **`uint16` Depolama:** `int64` veya ham metin yerine `uint16` kullanımı disk tüketimini ~750 MB'tan ~180 MB'a indirmiş ve G/Ç hızını artırmıştır.

---

## 6. Sıradaki Adım

Aşama 2 modelleme aşamasına geçiş:
- One-hot karakter kodlamasından LIF spike katmanına geçiş veya doğrudan Character Embedding $\to$ Leaky katmanı tasarımı (`CharSNNLM`).
- DirectML / CPU üzerinde otoregresif dil modeli eğitim döngüsü ve Perplexity (PPL) / BPC (Bits-per-character) metriklerinin izlenmesi.

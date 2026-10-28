# Mergen-SNN Sistem Mimarisi

Bu belge, **Mergen-SNN** projesinin donanım soyutlama, nöron dinamikleri, optimizasyon stratejileri ve dil modelleme yol haritasını özetler.

---

## 1. Temel Bileşenler ve Altyapı

| Bileşen | Seçilen Teknoloji / Yaklaşım | Açıklama |
| :--- | :--- | :--- |
| **Çerçeve** | PyTorch 2.4.1 (`torch==2.4.1+cpu`) | Kararlı temel derin öğrenme çatısı |
| **SNN Kütüphanesi** | snnTorch 1.0.0 | Biyolojik esintili nöromorfik katmanlar ve surrogate gradient desteği |
| **Donanım Hızlandırma** | DirectML (`torch-directml`) | Windows üzerinde AMD Radeon RX 580 (`privateuseone:0`) çalıştırma ortamı |
| **Optimizatör** | SGD (`foreach=False`, `momentum=0.9`) | DirectML'de CPU fallback oluşturmayan, saf donanım akışı |
| **Paket Yönetimi** | `uv` (Python 3.11) | Hızlı ve izole sanal ortam yönetimi |

---

## 2. Nöron Modeli: Leaky Integrate-and-Fire (LIF)

Projede temel hesaplama birimi olarak **Leaky Integrate-and-Fire (LIF)** modeli kullanılmaktadır:

$$U[t] = \beta U[t-1] + W X[t] - S[t-1] U_{\text{reset}}$$

- **Membran Bozunumu ($\beta$):** Varsayılan olarak $\beta = 0.9$ seçilmiştir.
- **Surrogate Gradient:** Geriye yayılım (BPTT) sırasında basamak fonksiyonunun türevsizliğini aşmak için snnTorch'un varsayılan düzgünleştirilmiş türev fonksiyonu kullanılır.
- **Ateşleme Dinamiği:** Membran potansiyeli eşik değeri aştığında $S[t]=1.0$ üretilir ve membran sıfırlanır.

---

## 3. Aşama 1: MNIST Doğrulama Mimarisi (Mevcut Durum)

Aşama 1'de SNN dinamikleri ve DirectML entegrasyonu MNIST veri seti üzerinde doğrulanmıştır:

```text
Girdi (28x28 = 784) 
      │
      ▼
Linear (784 → 100)
      │
      ▼
snn.Leaky (beta=0.9, num_steps=25)  ─── [Gizli Katman Spike'ları]
      │
      ▼
Linear (100 → 10)
      │
      ▼
snn.Leaky (beta=0.9, num_steps=25)  ─── [Çıkış Katmanı Spike'ları]
      │
      ▼
Rate Loss (Cross Entropy) & Rate Accuracy
```

- **Parametre Sayısı:** 79,510
- **Zaman Adımı:** $T = 25$
- **Kayıp Fonksiyonu:** `snntorch.functional.ce_rate_loss`
- **Doğruluk:** %93.80 (Adam) / %93.79 (SGD lr=0.01)

---

## 4. Aşama 2: Karakter Seviyesi Dil Modeli Veri Hattı (Stage 2 Data Pipeline)

Aşama 2 için 92.03 MB'lık 16 kategorili Bilgisayar Bilimleri (CS) korpusu karakter düzeyinde işlenmiş ve model eğitimine hazır hale getirilmiştir:

### 4.1. Karakter Kümesi Formatı (`charset.json`)
- **Filtreleme Eşiği:** `min_freq = 50` (nadir sembol ve matematik işaretleri ayıklanır).
- **Zorunlu Karakterler (`REQUIRED_CHARS`):** ASCII yazdırılabilir karakterler (32-126), 12 Türkçe karakter (`çÇğĞıİöÖşŞüÜ`) ve yaygın noktalama işaretleri (`‘’“”–—…«»`) frekansa bakılmaksızın her zaman sözlükte yer alır.
- **Bilinmeyen Karakter Tokeni (`<unk>`):** Sözlüğün ilk elemanı olarak `id = 0` değerine atanmıştır. Bilinmeyen veya sözlük dışı karakterler `encode()` sırasında `<unk>` (0) indeksine eşlenir.
- **Kelime Dağarcığı (Vocab Size):** 530 karakter (126 zorunlu + min_freq>=50 eklenen 413 + 1 `<unk>`).
- **Şema:**
  - `chars`: Alfabetik sıralı karakter listesi (`<unk>` başta).
  - `stoi`: Karakterden tamsayı indeksine eşleme (`{char: int}`).
  - `itos`: İndeksten karaktere eşleme (`{int: char}`).
  - `freq`: Karakter frekans istatistikleri.
- **Konumlar:** `data/processed/charset.json` ve `data/raw/tokenizers/charset.json` (eski v1 yedeği: `charset_v1.json.bak`).

### 4.2. İkili Token Depolama Formatı (`train.bin` ve `val.bin`)
- **Veri Tipi:** `np.uint16` (`vocab_size < 65,536` olduğu için bellek ve disk tasarrufu sağlar; 2 bayt/token).
- **Bölme Oranı:** %90 Eğitim (`train.bin`: 83,873,842 token, ~160 MB), %10 Doğrulama (`val.bin`: 9,319,316 token, ~18 MB).
- **G/Ç Simetrisi:** NumPy `np.save` ve `np.load` ile doğrudan dosya tanıtıcısı üzerinden okunup yazılır.

### 4.3. CharDataset Sözleşmesi ve Veri Besleme
- **Sınıf:** [`CharDataset`](file:///src/mergen/data/char_dataset.py) (`torch.utils.data.Dataset`).
- **Pencere Boyutu:** Sabit `block_size = 32`.
- **Otoregresif Hedef Kaydırma:** Bir sonraki karakteri tahmin etme kuralı:
  $$x = \text{data}[i : i + \text{block\_size}]$$
  $$y = \text{data}[i + 1 : i + \text{block\_size} + 1]$$
  Dolayısıyla her örnek için $y[k] = x[k+1]$ ($k < \text{block\_size}-1$) eşitliği sağlanır.

---

## 5. Nihai Hedef: Bilgisayar Bilimleri Odaklı SNN Dil Modeli

Mergen-SNN'in nihai amacı, standart tüketici donanımında (CPU / DirectML) son derece düşük enerji ve hesaplama maliyetiyle çalışan, Bilgisayar Bilimleri (CS) literatürü ve teknik dokümantasyon üzerinde uzmanlaşmış bir Spiking Dil Modeli (SLM) geliştirmektir.

- **Aşama 2:** Karakter düzeyinde (char-level), sabit bağlam pencereli Spiking RNN / LIF mimarisi.
- **Aşama 3:** Alt-kelime (Subword) tokenizer ve Transformer tabanlı SNN mimarileri (SpikeGPT / Spikformer yaklaşımları).

---

## 6. Gelecek Vizyonu: Klasik LLM Disiplini + SNN Verimliliği

Kendi Cython motorumuzla çalışan, BPE tokenizer kullanan, ~10M parametreli bir SNN dil modeli (**Mergen-Tiny**). Eğitimde klasik LLM disiplini (gradient clipping, LR warmup, checkpointing, BPB takibi) uygulanacak. RAG ve araçlarla güçlendirilecek.

### 6.1. Temel Yapıtaşları
1. **Özel Hesaplama Motoru (Cython):** Matris çarpımları (matmul) ve zaman adımlı sequential LIF membran güncellemeleri C-seviyesinde optimize edilerek Python döngü yükü sıfırlanacaktır.
2. **Byte-Level BPE Tokenizer:** 460 karakterlik sözlükten ~8K-16K alt-kelime (subword) sözlüğüne geçilerek bağlam penceresi genişletilecek; kod yapıları ve CS terimleri daha etkin temsil edilecektir.
3. **10M Parametre Ölçeği (Mergen-Tiny):** 275K kapasite sınırını aşarak sözdizimsel ve anlamsal temsili mümkün kılacak minimum ölçek.
4. **Klasik LLM Eğitim Disiplini:**
   - **Gradient Clipping:** SNN'lerde zamanla biriken surrogate gradyan patlamalarını önleme.
   - **Öğrenme Oranı Isınması (LR Warmup) & Cosine Decay:** Kararlı optimizasyon akışı.
   - **Bits-Per-Byte (BPB) & Cross-Entropy Takibi:** Karakter ve BPE seviyesinde adil kıyaslama metrikleri.
5. **RAG ve Araç Güçlendirmesi:** Hafif yerel vektör indeksi (CS dokümantasyonu, kılavuzlar) ve kod sözdizim denetleyicileri ile desteklenmiş çıkarım.

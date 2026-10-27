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

## 4. Nihai Hedef: Bilgisayar Bilimleri Odaklı SNN Dil Modeli

Mergen-SNN'in nihai amacı, standart tüketici donanımında (CPU / DirectML) son derece düşük enerji ve hesaplama maliyetiyle çalışan, Bilgisayar Bilimleri (CS) literatürü ve teknik dokümantasyon üzerinde uzmanlaşmış bir Spiking Dil Modeli (SLM) geliştirmektir.

- **Aşama 2:** Karakter düzeyinde (char-level), sabit bağlam pencereli Spiking RNN / LIF mimarisi.
- **Aşama 3:** Alt-kelime (Subword) tokenizer ve Transformer tabanlı SNN mimarileri (SpikeGPT / Spikformer yaklaşımları).

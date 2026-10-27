# Mimari Karar Günlüğü (Architecture Decision Records)

Bu belgede, Mergen-SNN projesinde alınan temel teknik kararlar, alternatifler ve seçim gerekçeleri kronolojik olarak kayıt altına alınmıştır.

---

## Karar Özeti Tablosu

| # | Karar Konusu | Alınan Karar | Karar Tarihi | Durum |
| :-: | :--- | :--- | :-: | :-: |
| **01** | **Donanım Arka Ucu & Optimizatör** | DirectML üzerinde varsayılan olarak **SGD (`momentum=0.9`, `foreach=False`)** kullanımı | 2026-10-27 | ✅ Kabul Edildi |
| **02** | **Karakter LM Girdi Temsili** | Aşama 2'ye **One-Hot → LIF** ile başlanması, ardından **Embedding → Spike**'a geçilmesi | 2026-10-27 | ✅ Kabul Edildi |
| **03** | **Bağlam ve Model Mimarisi** | Aşama 2'de **Sabit Pencereli Spiking RNN**, Aşama 3'te **SpikeGPT/Spikformer** mimarisi | 2026-10-27 | ✅ Kabul Edildi |

---

## Detaylı Karar Kayıtları

### ADR 01: DirectML Üzerinde SGD Optimizatörünün Seçilmesi
- **Bağlam:** Adam optimizatörü PyTorch 2.4.1 DirectML arka ucunda çalıştırıldığında `aten::lerp.Scalar_out` operatörünün desteklenmemesi nedeniyle her adımda CPU fallback uyarısı üretmekteydi. `foreach=False` yapıldığında bile PyTorch'un standart skaler Adam implementasyonu `exp_avg.lerp_` çağrısı yapmaya devam etmektedir.
- **Alternatifler:** 
  1. Adam ile devam etmek (her adımda CPU↔GPU bellek kopyalama maliyeti ve uyarılar).
  2. Özel bir C++/HLSL lerp operatörü derlemek.
  3. SGD (`momentum=0.9`, `foreach=False`) kullanmak.
- **Sonuç & Gerekçe:** SGD, DirectML üzerinde sıfır uyarı ile tamamen donanım hızlandırmalı çalışmış, epoch süresini 188.75s'den 179.64s'ye düşürmüş ve uygun öğrenme oranıyla (%93.79) Adam ile birebir aynı doğruluğu vermiştir. Aşama 1 için varsayılan optimizatör SGD yapılmıştır.

---

### ADR 02: Karakter Seviyesinde Girdi Kodlama Stratejisi
- **Bağlam:** Dil modellemesinde metin girdilerinin SNN nöronlarına nasıl besleneceği kritik bir tasarım kararıdır.
- **Alternatifler:**
  1. Doğrudan Poisson/Rate rate kodlaması.
  2. One-hot vektör → Doğrudan LIF akımı.
  3. Sürekli Embedding katmanı → Spike dönüştürücü.
- **Sonuç & Gerekçe:** Karakter düzeyinde (vocab boyutu ~100-256) hızlı prototipleme ve seyreklik avantajını doğrudan görmek için Aşama 2'de **One-Hot → LIF** akımıyla başlanacaktır. Kelime/alt-kelime düzeyine geçildiğinde (Aşama 3) **Embedding → Spike** mekanizmasına geçilecektir.

---

### ADR 03: Dil Modeli Bağlam ve Ağ Mimarisi Aşamalandırması
- **Bağlam:** SNN'lerde zaman boyutunun (temporal dynamics) dil modellemesinde nasıl kullanılacağı.
- **Aşamalandırma:**
  - **Aşama 2 (Hafif & Hızlı):** Sabit pencereli (fixed-window, ör. $L=64$ karakter) tekrarlayan (recurrent) LIF blokları. Doğrudan CPU ve RX 580 üzerinde eğitilebilir.
  - **Aşama 3 (Ölçekleme & Transformer):** Bilgisayar Bilimleri külliyatı için SpikeGPT veya Spikformer benzeri kendi kendine dikkat (self-attention) mekanizmasını spike dinamikleriyle değiştiren mimarilere geçiş.

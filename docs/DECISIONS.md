# Mimari Karar Günlüğü (Architecture Decision Records)

Bu belgede, Mergen-SNN projesinde alınan temel teknik kararlar, alternatifler ve seçim gerekçeleri kronolojik olarak kayıt altına alınmıştır.

---

## Karar Özeti Tablosu

| # | Karar Konusu | Alınan Karar | Karar Tarihi | Durum |
| :-: | :--- | :--- | :--- | :-: |
| **01** | **Donanım Arka Ucu & Optimizatör** | DirectML üzerinde varsayılan olarak **SGD (`momentum=0.9`, `foreach=False`)** kullanımı | 2026-10-27 | ✅ Kabul Edildi |
| **02** | **Karakter LM Girdi Temsili** | Aşama 2'ye **One-Hot → LIF** ile başlanması, ardından **Embedding → Spike**'a geçilmesi | 2026-10-27 | ✅ Kabul Edildi |
| **03** | **Bağlam ve Model Mimarisi** | Aşama 2'de **Sabit Pencereli Spiking RNN**, Aşama 3'te **SpikeGPT/Spikformer** mimarisi | 2026-10-27 | ✅ Kabul Edildi |
| **04** | **Stage 2 Donanım Seçimi: CPU vs DirectML** | Stage 2 model boyutunda (275K parametre), sequential LIF döngüsü için **CPU kullanımı** (8x daha hızlı) | 2026-10-28 | ✅ Kabul Edildi |
| **05** | **Klasik Transformer Yerine SNN Tercihi** | Donanım kısıtları (AMD RX 580 + CPU) ve enerji/hesaplama verimliliği nedeniyle **SNN mimarisinin seçilmesi** | 2026-10-28 | ✅ Kabul Edildi |
| **06** | **Model Ölçekleme: Mergen-Tiny (10M)** | 275K PoC modelindeki plato sonrası dil öğrenimi için minimum **~10M parametre ölçeğinin hedeflenmesi** | 2026-10-28 | ✅ Kabul Edildi |
| **07** | **Tokenizasyon Dönüşümü: Byte-Level BPE** | Char-level (460 vocab) yerine kod ve CS metinlerini temsil edecek **Byte-Level BPE Tokenizer'a geçilmesi (PDF B08)** | 2026-10-28 | ✅ Kabul Edildi |
| **08** | **Eğitim Güvenilirliği & Kesintisiz Çalışma** | Uzun süreli model eğitimlerinde **Step-Level Checkpointing (500 adım)** ve donanımsal **UPS (Kesintisiz Güç Kaynağı)** şartı | 2026-10-29 | ✅ Kabul Edildi |

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

---

### ADR 04: Aşama 2 İçin CPU Donanım Seçimi (DirectML Karşılaştırması)
- **Bağlam:** Aşama 2 karakter seviyesi SNN dil modeli (275K parametre, 32 timestep) eğitimi için DirectML ve CPU başa baş hız testine tabi tutulmuştur.
- **Bulgular & Ölçüm:** Stage 2 model boyutunda (275K parametre), CPU (61ms/step) DirectML'den (492ms/step) 8x daha hızlıdır.
- **Gerekçe:** 32 timestep sequential LIF döngüsü GPU'da paralelleştirilememekte, DirectML kernel launch overhead'i kazancı yutmaktadır. Bu nedenle Aşama 2 eğitiminde CPU tercih edilmiştir.
- **Gelecek Planı:** Stage 3'te model >5M parametreye ulaştığında ve matris boyutları büyüdüğünde DirectML tekrar test edilecektir.

---

### ADR 05: Donanım Kısıtları Nedeniyle Klasik Transformer Yerine SNN Tercihi
- **Bağlam:** Tüketici sınıfı donanım (AMD Radeon RX 580 8GB + çok çekirdekli CPU) ortamında standart yoğun Transformer modelleri (özellikle $O(L^2)$ öz-dikkat matrisleri ve yoğun aktivasyonlar) yüksek bellek ve hesaplama maliyeti yaratmaktadır.
- **Alternatifler:**
  1. Klasik yoğun Transformer (GPT-2, LLaMA-mini).
  2. Seyrek Dikkat (Sparse Attention / Linformer).
  3. Biyolojik esintili Spiking Neural Network (SNN).
- **Sonuç & Gerekçe:** SNN'lerin olay tabanlı (event-driven) seyrek ateşleme dinamiği, çarpma yerine toplama (MAC yerine AC) yapısı ve zaman boyutundaki hafıza yeteneği; yerel donanımda yüksek enerji/hesaplama verimliliği sunar. Bu nedenle proje genelinde SNN omurgası esas alınmıştır.

---

### ADR 06: Model Ölçekleme: Mergen-Tiny (~10M Parametre) Hedefi
- **Bağlam:** Aşama 2'de 275K parametreli `CharSNNLM` modeli 93M token üzerinde 102.756 adım eğitilmiş, ancak Train loss 2.56 ve Val loss 2.66 seviyesinde plato yapmıştır. Model kelime köklerini ve noktalama biçimlerini öğrenmesine rağmen kapasite yetersizliği sebebiyle anlamlı cümleler kuramamıştır.
- **Sonuç & Gerekçe:** Ampirik bulgular, doğal dilin sözdizimsel ve anlamsal hiyerarşisini öğrenebilmek için model kapasitesinin minimum **~10M parametre (Mergen-Tiny)** seviyesine çıkarılması gerektiğini göstermiştir. Aşama 3 bu ölçekleme hedefiyle yürütülecektir.

---

### ADR 07: Karakter Düzeyinden Byte-Level BPE Tokenizer'a Geçiş (PDF B08)
- **Bağlam:** Aşama 2'de kullanılan karakter seviyesi tokenizasyon (460 vocab), `block_size = 32` penceresinde yalnızca 5-6 kelimelik çok dar bir bağlam aralığı sunmakta ve modelin uzun menzilli CS kavramlarını öğrenmesini engellemektedir.
- **Alternatifler:**
  1. Karakter düzeyinde bağlam penceresini $L=256$'ya çıkarmak (hesaplama maliyeti $8\times$ artar).
  2. Kelime düzeyinde tokenizer (büyük sözlük boyutu ve OOV problemleri).
  3. Byte-Level BPE (Byte-Pair Encoding) Tokenizer (PDF B08 spesifikasyonu).
- **Sonuç & Gerekçe:** Byte-Level BPE; Python/C++ kod sözdizimini, özel anahtar kelimeleri ve teknik İngilizce terimleri sıkıştırılmış alt-kelime dizileri halinde temsil eder. Bu geçiş, aynı bağlam uzunluğunda modele yaklaşık 4-5 kat daha geniş bir anlamsal görüş açısı kazandıracaktır.

---

### ADR 08: Uzun Eğitimler İçin Kesintisiz Güç Kaynağı (UPS) ve Step-Level Checkpoint Şartı
- **Bağlam:** Aşama 3 ölçeğinde (6.6M parametre, 31K adım, ~3-4 saat) gece ve çoklu epoch koşularında elektrik kesintileri, sistem yeniden başlatmaları veya uyku modu geçişleri gibi dış etkenler eğitim ilerlemesini kaybetme riski taşır.
- **Alternatifler:**
  1. Yalnızca epoch sonunda checkpoint kaydetmek (1 epoch kesintisinde 35-45 dakikalık tüm ilerleme kaybolur).
  2. Her adımda checkpoint kaydetmek (I/O darboğazı ve disk şişmesi).
  3. Her 500 adımda bir dönerli (rolling, son 3 adet) step-level checkpoint kaydetmek.
- **Sonuç & Gerekçe:**
  1. Kod seviyesinde her 500 adımda bir `checkpoints/bpe_lm_step{step}.pt` checkpoint'i alınır; disk tasarrufu için daima son 3 checkpoint saklanır.
  2. `train_lm.py` script'ine `--resume` argümanı ve `resume: auto` desteği eklenerek kesinti sonrası en son adımdan şeffaf biçimde devam edilmesi sağlanmıştır.
  3. Donanım seviyesinde uzun süreli eğitim koşuları için **UPS (Kesintisiz Güç Kaynağı)** kullanımı şart olarak not düşülmüştür.

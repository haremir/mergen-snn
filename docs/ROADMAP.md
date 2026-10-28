# Mergen-SNN Geliştirme Yol Haritası (Roadmap)

Mergen-SNN projesinin hedefleri dört ana aşamada yürütülmektedir:

---

## 🗺️ Aşama Takvimi

```text
[Aşama 1: Doğrulama] ───► [Aşama 2: Char-LM PoC] ───► [Aşama 3: Ölçekleme & Optimizasyon] ───► [Aşama 4: Metrik & RAG]
       ✅ Tamamlandı              ✅ Tamamlandı                      ⏳ Sırada                         📋 Planlandı
```

---

### Aşama 1: SNN Temelleri ve Doğrulama ✅ (Tamamlandı)
- [x] Temel mimari ve `uv` bağımlılık altyapısının kurulması.
- [x] DirectML (`torch-directml`) arka ucu ile AMD Radeon RX 580 entegrasyonu.
- [x] MNIST veri yükleyici ve döngüsüz Poisson/Rate kodlama fonksiyonları.
- [x] İki katmanlı SNN LIF mimarisi (`SNNMNIST`, 79,510 parametre).
- [x] Cross-entropy rate loss ve rate accuracy metrikleri.
- [x] Optimizer optimizasyonu (Adam vs SGD karşılaştırması ve CPU fallback'siz temiz akış).
- [x] **Sonuç:** 1 epoch'ta **%93.80** test doğruluğu elde edildi.

---

### Aşama 2: Karakter Seviyesinde SNN PoC ✅ (Tamamlandı)
- [x] 93 MB'lık 16 kategorili CS korpusunun indirilmesi ve temizlenmesi (`prepare_data.py`).
- [x] C0/C1 kontrol karakterlerinin ayıklanması, ligatür normalizasyonu, Türkçe karakter garantili 460 vocab'lık `charset.json` üretimi.
- [x] Sabit bağlam pencereli `CharDataset` ve token akışı boru hattı.
- [x] 275K parametreli `CharSNNLM` mimarisi (Embedding → 2x LIF → FC).
- [x] CPU üzerinde tam külliyat eğitimi (~71 dakika, 102.756 adım).
- [x] Top-k ($k=40$) ve Top-p ($p=0.9$) sıcaklık örneklemeli `generate.py` arayüzü ve REPL modu.
- [x] **Sonuç:** Train loss 2.56, val loss 2.66 (PPL 13.35). Model kelime köklerini (`ing`, `the`, `func`, `there`) öğrendi; kapasite limiti (275K) ve plato ampirik olarak doğrulandı.

---

### Aşama 3: Ölçekleme ve Optimizasyon (Mergen-Tiny) ⏳ (Sıradaki Aşama)
- [ ] **BPE Tokenizer:** Char-level (460 vocab) yerine kod ve CS metinlerini temsil edecek Byte-Level BPE Tokenizer'a geçiş (`data/tokenizers/`).
- [ ] **10M Model Ölçeklemesi (Mergen-Tiny):** Model parametre kapasitesinin ~10M parametreye çıkarılması.
- [ ] **Klasik LLM Eğitim Disiplini:** Gradient clipping, LR warmup, cosine decay öğrenme oranı çizelgeleyici, checkpointing ve BPB (Bits-Per-Byte) takibi.
- [ ] **Cython Hesaplama Motoru:** Matris çarpımı (matmul) ve sequential LIF membran güncelleme döngülerinin Cython C-uzantıları (`src/mergen/cython_ext/`) ile hızlandırılması.
- [ ] **Donanım Değerlendirmesi:** Büyüyen matris boyutlarıyla DirectML GPU hızlandırmasının (>5M param) tekrar test edilmesi.

---

### Aşama 4: Metrik Standardizasyonu ve RAG Entegrasyonu 📋 (Planlandı)
- [ ] **Seyreklik ve Enerji Metrikleri:** Nöron ateşleme seyreklik oranlarının (sparsity ratio) ve teorik nöromorfik enerji tüketiminin ölçülmesi.
- [ ] **RAG Altyapısı:** CS dokümantasyonu, man sayfaları ve teknik kitaplar için hafif yerel vektör dizini (Retrieval-Augmented Generation).
- [ ] **Araç ve Fonksiyon Entegrasyonu:** Kod sözdizimi denetimi ve komut çalıştırma araçlarıyla modelin zenginleştirilmesi.
- [ ] **Karşılaştırma (Benchmark):** Standart küçük ANN/Transformer modelleriyle (NanoGPT, Tiny-LLaMA) doğruluk, enerji ve çıkarım hızı kıyaslamaları.

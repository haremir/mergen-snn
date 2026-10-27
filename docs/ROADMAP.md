# Mergen-SNN Geliştirme Yol Haritası (Roadmap)

Mergen-SNN projesinin hedefleri dört ana aşamada yürütülmektedir:

---

## 🗺️ Aşama Takvimi

```text
[Aşama 1: Doğrulama] ───► [Aşama 2: Char-LM] ───► [Aşama 3: CS Korpusu & SpikeGPT] ───► [Aşama 4: Ölçekleme & Cython]
       ✅ Tamamlandı             ⏳ Sırada                      📋 Planlandı                         📋 Planlandı
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

### Aşama 2: Karakter Düzeyinde Spiking Dil Modeli ⏳ (Sıradaki Aşama)
- [ ] Karakter seviyesinde metin veri yükleyicisi (Shakespeare / mini-CS metinleri).
- [ ] Girdi temsili: Sabit bağlam pencereli (ör. $L=64$) One-Hot → LIF kodlaması.
- [ ] Tekrarlayan Spiking Sinir Ağı (Spiking RNN / Recurrent LIF) mimarisi.
- [ ] Bir sonraki karakter tahmin kaybı (Next-token prediction cross-entropy).
- [ ] Sıcaklık (temperature) parametreli spiking metin üretim (sampling) arayüzü.

---

### Aşama 3: CS Uzmanlaşma ve Transformer Tabanlı SNN 📋 (Planlandı)
- [ ] Bilgisayar Bilimleri külliyatının indirilmesi ve ön işlenmesi (`scripts/fetch_corpus.py`).
- [ ] BPE / WordPiece alt-kelime (subword) tokenizer eğitimi (`data/tokenizers/`).
- [ ] Embedding → Zaman adımı spike dönüştürücü modülü.
- [ ] Spikformer / SpikeGPT benzeri spike tabanlı dikkat (Spiking Self-Attention) blokları.
- [ ] CS kavram ve kod blokları üzerinde üretim kalitesinin ölçülmesi.

---

### Aşama 4: Değerlendirme, Seyreklik ve Cython Uzantıları 📋 (Planlandı)
- [ ] Modelin nöron ateşleme seyreklik oranlarının (sparsity ratio) ve teorik enerji tüketiminin ölçülmesi.
- [ ] Kritik nöron simülasyon döngüleri için Cython C-extension optimizasyonu (`src/mergen/cython_ext/`).
- [ ] CPU vs DirectML vs Cython çıkarım (inference) hız kıyaslamaları.
- [ ] Standart küçük ANN modelleriyle (LSTM, GPT-mini) enerji/performans karşılaştırması.

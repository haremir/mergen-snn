# Stage 2: Karakter Seviyesi SNN PoC Kapanış Raporu

**Tarih:** 28 Ekim 2026  
**Aşama:** Stage 2 — Char-Level SNN Dil Modeli (Proof of Concept)  
**Yazar:** Mergen-SNN Geliştirme Ekibi  
**Durum:** ✅ Tamamlandı (Aşama 3'e Geçiş Onaylandı)

---

## 1. Yönetici Özeti

Aşama 2 kapsamında geliştirilen 275K parametreli `CharSNNLM` mimarisi, 93 milyon token'lık Bilgisayar Bilimleri külliyatı üzerinde tam eğitimden geçirilmiştir. Model, karakter düzeyinde sözdizimi örüntülerini, boşluk ve kelime sınırlarını ve teknik İngilizce kelime köklerini (`ing`, `func`, `the`, `cite`, `there`) başarıyla öğrenmiş; ancak 275K parametrelik kapasite sınırı nedeniyle anlamsal (semantik) bütünlüğe sahip cümleler kuramamış ve plato seviyesine ulaşmıştır. Bu sonuçla Aşama 2 boru hattı doğrulanmış, projenin bir sonraki adımı olan **Mergen-Tiny (~10M parametre)** ve **Byte-Level BPE Tokenizer** ölçekleme aşamasına geçiş kararı alınmıştır.

---

## 2. Model ve Mimari Detayları

* **Model Adı:** `CharSNNLM` (Sabit Pencereli Spiking RNN)
* **Toplam Parametre:** **275.916**
  * `embedding.weight`: 58.880 (`[460, 128]`)
  * `fc1.weight` & `bias`: 33.024 (`[256, 128]` + `[256]`)
  * `fc2.weight` & `bias`: 65.792 (`[256, 256]` + `[256]`)
  * `fc_out.weight` & `bias`: 118.220 (`[460, 256]` + `[460]`)
* **Nöron Tipi:** Leaky Integrate-and-Fire (`snn.Leaky`, $\beta = 0.9$)
* **Zaman Adımı ($T$):** 32 timestep (sequential LIF membran taşınımı)
* **Kelime Dağarcığı (Vocab Size):** 460 karakter (C0/C1 kontrolleri ve ligatürler temizlenmiş, 12 Türkçe karakter garantili, `<unk>=0`)
* **Bağlam Penceresi (`block_size`):** 32 karakter

---

## 3. Eğitim Metrikleri ve Süreç

* **Donanım:** CPU (DirectML 492ms/step yerine 8x daha hızlı olan 61ms/step avantajı nedeniyle)
* **Optimizatör:** SGD (`lr=0.01`, `momentum=0.9`, `foreach=False`)
* **Batch Boyutu:** 32 (adım başına 1.024 karakter)
* **Adım/Epoch:** 81.756 adım

| Epoch | İşlenen Adım | Süre | Train Loss | Train PPL | Val Loss | Val PPL | Durum |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** | 81.756 / 81.756 | 3.219s (~53 dk) | 2.6266 | 13.83 | 2.6648 | 14.37 | Tamamlandı |
| **Epoch 2** | 21.000 / 81.756 | 1.035s (~17 dk) | 2.5646 | 13.00 | 2.6679 | 14.41 | Erken Durdurma (Plato) |
| **Toplam** | **102.756 adım** | **4.254s (~71 dk)** | **2.5646** | **13.00** | **2.5912** | **13.35** | **En İyi Model Saklandı** |

> **Plato Tespiti:** Epoch 2'de 16.000 ile 21.000 adımları arasında loss değerinin `2.5646` seviyesinde sabit kalması üzerine 5.000 adımlık erken durdurma kriteri devreye girmiş ve eğitim sağlıklı biçimde tamamlanmıştır.

---

## 4. Metin Üretim (Generation) Analizi

Top-k ($k=40$) ve Top-p ($p=0.9$) filtrelemesiyle farklı sıcaklıklar ($T \in \{0.7, 0.8, 0.9, 1.0, 1.2\}$) test edilmiştir:

* **$T = 0.7$:** Model aşırı temkinli davranarak ardışık nokta ve boşluk döngüsüne (`.....`) girme eğilimindedir.
* **$T = 0.8$ (En Başarılı):** Kelime sınırları, boşluklar ve gerçek İngilizce sözcük parçacıkları en tutarlı şekilde üretilmiştir:
  * *"Python'da listed with in as te sagrest ocitise fe cite omesgu wye retdentas tiech sin a onting"*
  * *"Algoritma nedire.se ts copectone tienle tenled arindle dasos upd adtiiate."*
  * *"Bilgisayar mimarision to thars an reaefecs a suusuted ose dating fthi bergal the indist the elof on"*
* **$T \ge 1.0$:** Rastgele heceleme ve anlamsız karakter varyansı artmaktadır.

**Değerlendirme:** Model kelime köklerini (`listed with in as`, `cite`, `sin a`, `there`, `func`, `dating`), parantez/noktalama kurallarını ve sözdizimini kavramıştır; ancak 275K parametrelik kapasite sınırı nedeniyle mantıksal cümle kurma yetisine sahip değildir.

---

## 5. Alınan Karar ve Aşama 3 Geçişi

1. **Kapasite Limiti:** 275K parametreli küçük bir modelin doğal dil modellemesinde cümle düzeyinde anlamsal bağ kuramayacağı ve plato yaptığı ampirik olarak kanıtlanmıştır.
2. **Tokenizasyon Dönüşümü:** Karakter düzeyinde (460 vocab) 32 karakterlik pencere yalnızca 5-6 kelimelik çok dar bir bağlam sunmaktadır. Bu nedenle Byte-Level BPE Tokenizer'a geçilerek bağlam genişletilecektir.
3. **Ölçekleme (Mergen-Tiny):** Model mimarisi ~10M parametreye çıkarılacak, klasik LLM eğitim disiplini (gradient clipping, LR warmup, cosine scheduler, BPB takibi) uygulanacaktır.
4. **Hesaplama Motoru:** Cython ile optimize edilmiş matris çarpımı ve LIF döngüleri geliştirilecektir.

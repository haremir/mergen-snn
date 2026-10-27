# Aşama 1: MNIST SNN Doğrulama Raporu

- **Tarih:** 27 Ekim 2026
- **Görev:** İki katmanlı SNN (LIF) modelinin MNIST üzerinde eğitimi ve DirectML doğrulamasının yapılması.
- **Hedef:** 1 epoch sonunda %90+ test doğruluğu elde etmek.

---

## 1. Model ve Eğitim Parametreleri

- **Model Sınıfı:** `SNNMNIST` (FC1 784→100 + LIF1, FC2 100→10 + LIF2)
- **Toplam Eğitilebilir Parametre:** 79,510
- **Zaman Adımı Sayısı ($T$):** 25
- **Membran Bozunumu ($\beta$):** 0.9
- **Donanım Arka Ucu:** DirectML (`privateuseone:0` — AMD Radeon RX 580)
- **Batch Boyutu:** 128 (469 adım / epoch)
- **Kayıp Fonksiyonu:** `snnTorch.functional.ce_rate_loss`

---

## 2. Sonuçlar Özeti

| Deney | Optimizatör | Öğrenme Oranı ($lr$) | Epoch Süresi | Train Loss / Acc | Test Loss / Acc | Konsol Uyarısı |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Deney 1 (İlk Koşu)** | Adam (varsayılan) | 0.001 | 189.05s | 1.6081 / %87.66 | 1.5307 / **%93.65** | `aten::lerp` uyarısı |
| **Deney 2 (foreach=False)** | Adam (`foreach=False`) | 0.001 | 188.75s | 1.5983 / %89.17 | 1.5301 / **%93.80** | `aten::lerp` uyarısı |
| **Deney 3 (SGD 10x)** | SGD (`momentum=0.9`) | 0.010 | 183.27s | 1.5758 / %89.64 | 1.5252 / **%93.79** | **Temiz (Uyarı yok)** |
| **Deney 4 (SGD 1x)** | SGD (`momentum=0.9`) | 0.001 | 179.64s | 1.7949 / %71.94 | 1.6553 / **%83.07** | **Temiz (Uyarı yok)** |

---

## 3. Bulgular ve Çıkarımlar

1. **Hedef Başarımı:** Hem Adam hem de SGD ($lr=0.01$) ile 1 epoch içerisinde **%93.8** seviyesine ulaşılarak %90'lık Aşama 1 hedefi rahatlıkla aşılmıştır.
2. **DirectML Uyumu:** Modelin ileri (forward) ve surrogate gradient geri yayılımı (backward) DirectML üzerinde sıfır hata ile çalışmıştır.
3. **Optimizatör Dinamiği:** SGD ile Adam'ın ürettiği `aten::lerp` CPU fallback uyarısı tamamen çözülmüş ve saf GPU akışı elde edilmiştir.

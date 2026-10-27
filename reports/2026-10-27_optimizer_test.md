# DirectML Optimizatör Karşılaştırma Raporu (Adam vs SGD)

- **Tarih:** 27 Ekim 2026
- **Amaç:** DirectML üzerinde PyTorch Adam optimizatörünün tetiklediği `aten::lerp.Scalar_out` CPU fallback uyarısının incelenmesi ve SGD alternatifiyle karşılaştırılması.

---

## 1. Karşılaştırma Tablosu

| Metrik / Özellik | Adam (`foreach=False`) | SGD (`lr=0.010`, momentum=0.9) | SGD (`lr=0.001`, momentum=0.9) |
| :--- | :---: | :---: | :---: |
| **Epoch Süresi** | 188.75 saniye | 183.27 saniye | **179.64 saniye** |
| **Eğitim Doğruluğu** | %89.17 | %89.64 | %71.94 |
| **Test Doğruluğu** | **%93.80** | **%93.79** | %83.07 |
| **DirectML Uyarı Durumu** | ⚠️ `aten::lerp` CPU fallback | ✅ **Uyarı Yok (Saf GPU)** | ✅ **Uyarı Yok (Saf GPU)** |
| **CPU↔GPU Transferi** | Her optimizer adımında var | **Yok** | **Yok** |

---

## 2. Teknik İnceleme

### Adam ve `aten::lerp` Darboğazı
PyTorch `torch.optim.Adam` kaynak kodunda moment kestirimleri şu şekilde güncellenmektedir:
```python
exp_avg.lerp_(grad, 1 - beta1)
```
DirectML eklentisi (`torch-directml`) `aten::lerp` operatörünü yerel olarak desteklemediği için tensörleri perde arkasında CPU'ya taşımakta, işlemi CPU üzerinde gerçekleştirmekte ve geri kopyalamaktadır. Bu durum her adımda `UserWarning` fırlatmakta ve küçük de olsa ek gecikme yaratmaktadır.

### SGD ile Çözüm
SGD (`momentum=0.9`) optimizatöründe kullanılan temel tensör işlemleri (`add_`, `mul_`) DirectML tarafından eksiksiz desteklenmektedir. 
- `lr=0.01` seçildiğinde (Adam'a kıyasla 10 kat), SGD tek epoch'ta **%93.79** test doğruluğuna ulaşarak Adam ile birebir aynı doğruluğu vermiştir.
- Süre olarak da epoch başına ~5.5 saniye kazanım sağlamıştır.

---

## 3. Alınan Karar
Aşama 1 için varsayılan optimizatör olarak **SGD** seçilmiştir. İlerleyen aşamalarda $lr=0.01$ (veya config üzerinden ayarlanabilir öğrenme oranı) ile SGD kullanılması DirectML'de temiz ve transfer maliyetsiz bir eğitim akışı sağlamaktadır.

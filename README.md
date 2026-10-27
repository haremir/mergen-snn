# Mergen-SNN

> Türk mitolojisindeki akıl, zeka ve bilgelik tanrısı **Mergen**'den esinlenen **Mergen-SNN**, biyolojik esintili ve enerji verimli Spiking Neural Network (SNN) mimarilerini modern dil modelleme yöntemleriyle bir araya getiren açık kaynaklı bir yapay zeka projesidir.

---

## 🎯 Proje Hedefi

Mergen-SNN, Bilgisayar Bilimleri (*Computer Science*) alanına odaklanan, standart tüketici donanımlarında (özellikle CPU üzerinde) verimli çalışabilen, modüler ve genişletilebilir bir Spiking Dil Modeli (Spiking Neural Network Language Model) geliştirmeyi hedefler. Geleneksel yapay sinir ağlarının yüksek enerji ve hesaplama maliyetine alternatif olarak nöromorfik ve seyreklik (sparsity) tabanlı hesaplama yaklaşımlarını keşfeder.

---

## 🗺️ Geliştirme Aşamaları

| Aşama | Başlık | Açıklama | Durum |
| :--- | :--- | :--- | :---: |
| **Aşama 0** | **İskelet & Altyapı** | Proje mimarisi, uv yapılandırması, CPU PyTorch & snnTorch entegrasyonu | ✅ Tamamlandı |
| **Aşama 1** | **SNN Temelleri & Doğrulama** | Sızıntılı Entegre-Ateşle (LIF) nöron modelleri, snnTorch doğrulaması ve MNIST testi | ⏳ Planlandı |
| **Aşama 2** | **Karakter Seviyesinde LM** | Karakter düzeyinde spiking RNN/Transformer tabanlı dil modelleme ve metin üretimi | ⏳ Planlandı |
| **Aşama 3** | **Tokenizer & CS Korpusu** | Alt-kelime (Subword) tokenizer ve Bilgisayar Bilimleri literatürü ile eğitim | ⏳ Planlandı |
| **Aşama 4** | **Ölçekleme & Hızlandırma** | Cython/C++ uzantıları, seyreklik optimizasyonları ve CPU kıyaslamaları | ⏳ Planlandı |

---

## 🚀 Kurulum

Proje, modern ve hızlı Python paket yöneticisi olan [`uv`](https://docs.astral.sh/uv/) ile yönetilmektedir.

### Ön Gereksinimler

- **Python**: 3.11
- **uv**: Paket yöneticisi ([Kurulum Rehberi](https://docs.astral.sh/uv/getting-started/installation/))

### Adımlar

1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/haremir/mergen-snn.git
   cd mergen-snn
   ```

2. Sanal ortamı oluşturun ve tüm bağımlılıkları (CPU PyTorch dahil) yükleyin:
   ```bash
   uv sync
   ```

3. Kurulumu doğrulayın:
   ```bash
   uv run python -c "import snntorch, torch; print(f'PyTorch: {torch.__version__} | snnTorch: {snntorch.__version__}')"
   ```

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) kapsamında lisanslanmıştır. Detaylar için LICENSE dosyasına bakabilirsiniz.

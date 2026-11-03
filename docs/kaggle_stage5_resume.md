# Kaggle Stage 5 v3 Resume Rehberi

## A) Kaggle Hesabı
- **Kayıt:** [kaggle.com](https://www.kaggle.com) → **Register with Google**
- **Doğrulama:** **Settings** → **Phone Verification** (30 saat haftalık GPU kotası için zorunlu)

## B) Notebook Ayarları
- **Oluştur:** **Create** → **New Notebook**
- **Settings Paneli (Sağ):**
  - **Accelerator:** `GPU T4 x2`
  - **Internet:** `On`
  - **Save**

## C) Dosya Yükleme ("Add Input" → "Upload")
Yerel diskten şu 3 dosyayı Kaggle'a dataset olarak yükleyin:
- `data/processed/tokenizer_bpe_v2.json`
- `cs_corpus_clean.tar.gz` (34 MB) — *(Yerel: `data/raw/cs_corpus_clean.tar.gz`)*
- `checkpoints/stage5_21m_step58000.pt` (262 MB)

## D) Notebook Hücreleri

### Hücre 1: Repoyu Klonla ve Ortamı Kur
```python
!git clone https://github.com/haremir/mergen-snn.git
%cd mergen-snn
!pip install -q -e .
!pip install -q snntorch pyyaml rich numpy datasets
```

### Hücre 2: Yüklenen Dosyaları Yerleştir ve Korpusu Aç
```python
import os, shutil, glob
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
for pattern, dest in [
    ('/kaggle/input/**/tokenizer_bpe_v2.json', 'data/processed/'),
    ('/kaggle/input/**/cs_corpus_clean.tar.gz', 'data/raw/'),
    ('/kaggle/input/**/stage5_21m_step58000.pt', 'checkpoints/'),
]:
    for f in glob.glob(pattern, recursive=True):
        shutil.copy(f, dest)
        print(f"OK: {f}")
!cd data/raw && tar -xzf cs_corpus_clean.tar.gz
# Sonra klasör adını düzelt (temiz kopya kullanılacak)
!mv data/raw/cs_corpus data/raw/cs_corpus_old 2>/dev/null || true
!mv data/raw/cs_corpus_kaggle data/raw/cs_corpus
```

### Hücre 3: Veri Hazırlama (v3 Dengeli Karışım)
```python
!python scripts/prepare_bpe_data_massive.py
```
*(Kaggle fiber interneti ile ~3-4 dakikada 936M token'lık `bpe_train_v3.bin` üretilir)*

### Hücre 4: Eğitimi Başlat / Resume Et
```python
!python scripts/train_lm.py --config configs/stage5_21m_16k_v3_kaggle.yaml --resume auto
```
*(Not: `early_stop_patience: 0` olarak ayarlanmıştır, plato kesintisi olmadan 1 tam epoch - 109.809 adım tamamlanır)*

### Hücre 5: Metin Üretim Testi
```python
!python scripts/generate.py --checkpoint checkpoints/stage5_21m_final.pt --prompt "Python'da liste"
!python scripts/generate.py --checkpoint checkpoints/stage5_21m_final.pt --prompt "def train("
!python scripts/generate.py --checkpoint checkpoints/stage5_21m_final.pt --prompt "Algoritma nedir?"
```

## E) Model İndirme
- Eğitim tamamlandığında sağ paneldeki **Output** sekmesini yenileyin.
- `/kaggle/working/mergen-snn/checkpoints/` altındaki `stage5_21m_final.pt` ve `stage5_21m_best.pt` dosyalarını indirin.

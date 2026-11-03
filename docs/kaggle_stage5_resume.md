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
- `cs_corpus.tar.gz` (34 MB)
- `checkpoints/stage5_21m_step58000.pt` (262 MB)

## D) Notebook Hücreleri

### Hücre 1
```python
!git clone https://github.com/haremir/mergen-snn.git
%cd mergen-snn
!pip install -q snntorch pyyaml rich numpy datasets
```

### Hücre 2
```python
import os, shutil, glob
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
for pattern, dest in [
    ('/kaggle/input/**/tokenizer_bpe_v2.json', 'data/processed/'),
    ('/kaggle/input/**/cs_corpus.tar.gz', 'data/raw/'),
    ('/kaggle/input/**/stage5_21m_step58000.pt', 'checkpoints/'),
]:
    for f in glob.glob(pattern, recursive=True):
        shutil.copy(f, dest)
        print(f"OK: {f}")
!cd data/raw && tar -xzf cs_corpus.tar.gz
```

### Hücre 3
```python
!python scripts/prepare_bpe_data_massive.py
```

### Hücre 4
```python
!python scripts/train_lm.py --config configs/stage5_21m_16k_v3_kaggle.yaml --resume auto
```

### Hücre 5
```python
!python scripts/generate.py --checkpoint checkpoints/stage5_21m_final.pt --prompt "Python'da liste"
!python scripts/generate.py --checkpoint checkpoints/stage5_21m_final.pt --prompt "def train("
!python scripts/generate.py --checkpoint checkpoints/stage5_21m_final.pt --prompt "Algoritma nedir?"
```

## E) Model İndirme
- Eğitim sonrası sağ üst **Output** panelinden `/kaggle/working/mergen-snn/checkpoints/` altındaki `stage5_21m_final.pt` ve `stage5_21m_best.pt` dosyalarını indirin.

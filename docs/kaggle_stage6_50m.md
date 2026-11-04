# Kaggle Stage 6 (50M SNN) Eğitim Rehberi

Bu rehber, 50M parametreli Mergen-SNN modelinin Kaggle T4 GPU üzerinde eğitilmesi için gerekli 5 hücreyi içerir.

---

### Hücre 1 — Kurulum ve Veri Hazırlığı
```python
import os, shutil, glob
from pathlib import Path

# Repo klonlama ve güncelleme
if not os.path.exists('/kaggle/working/mergen-snn'):
    !git clone https://github.com/haremir/mergen-snn.git /kaggle/working/mergen-snn
%cd /kaggle/working/mergen-snn
!git pull

# Bağımlılıklar (torch 2.4.1 pinini kaldır, Kaggle'ın kendi torch'u kullanılsın)
!pip install -q snntorch pyyaml rich numpy datasets tokenizers
!pip install -q -e . --no-deps

# Input'tan dosyaları kopyala
os.makedirs('data/processed', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
for f in glob.glob('/kaggle/input/**/tokenizer_bpe_v2.json', recursive=True):
    shutil.copy2(f, 'data/processed/')
for f in glob.glob('/kaggle/input/**/bpe_train_v3.bin', recursive=True):
    shutil.copy2(f, 'data/processed/')
for f in glob.glob('/kaggle/input/**/bpe_val_v3.bin', recursive=True):
    shutil.copy2(f, 'data/processed/')

# Doğrula
!ls -lh data/processed/
```

---

### Hücre 2 — Config Oluşturma
```python
import yaml, os
os.makedirs('configs', exist_ok=True)
cfg = {
    'data': {'block_size': 128, 'batch_size': 32,
             'train_bin': 'data/processed/bpe_train_v3.bin',
             'val_bin': 'data/processed/bpe_val_v3.bin'},
    'model': {'vocab_size': 16384, 'embed_dim': 768, 'hidden_size': 768,
              'num_steps': 32, 'beta': 0.95, 'threshold': 1.0,
              'tie_weights': False, 'use_cython': False},
    'training': {'epochs': 1, 'learning_rate': 0.0003, 'optimizer': 'adam',
                 'warmup_steps': 500, 'clip_grad_norm': 1.0,
                 'checkpoint_every': 2000, 'resume': 'auto',
                 'checkpoint_dir': 'checkpoints', 'device': 'cuda'}
}
with open('configs/stage6_50m_16k.yaml', 'w') as f:
    yaml.dump(cfg, f)
print("Config hazır.")
```

---

### Hücre 3 — Eğitim Başlatma (Blocking, ~5-6 saat)
```python
!mkdir -p runs
!python scripts/train_lm.py --config configs/stage6_50m_16k.yaml 2>&1 | tee runs/stage6_50m_train.log
```

---

### Hücre 4 — Checkpoint'leri Output'a Kopyalama (Eğitim bittikten sonra)
```python
import shutil, glob
os.makedirs('/kaggle/working/output_checkpoints', exist_ok=True)
for f in glob.glob('checkpoints/stage6_50m_*.pt'):
    shutil.copy2(f, '/kaggle/working/output_checkpoints/')
    print(f"Kopyalandı: {f}")
print("Output panelinden indirebilirsin.")
```

---

### Hücre 5 — Resume (Oturum koptuysa yeniden başlatıldığında)
```python
!python scripts/train_lm.py --config configs/stage6_50m_16k.yaml --resume auto 2>&1 | tee -a runs/stage6_50m_train.log
```

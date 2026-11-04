# Kaggle Stage 6 (Gerçek 50M SNN + 2B Token) Eğitim Rehberi

Bu rehber, **49.8M parametreli** gerçek Mergen-SNN modelinin ve **2B tokenlık** veri kümesinin (`bpe_train_v3b.bin`) Kaggle GPU (T4 / T4 x2) üzerinde eğitilmesi için gerekli 5 hücreyi içerir.

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

# Bağımlılıklar (Kaggle PyTorch'u korunarak snntorch kurulur)
!pip install -q snntorch pyyaml rich numpy datasets tokenizers
!pip install -q -e . --no-deps

# Input'tan 2B tokenlık verileri ve tokenizer'ı kopyala
os.makedirs('data/processed', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
for f in glob.glob('/kaggle/input/**/tokenizer_bpe_v2.json', recursive=True):
    shutil.copy2(f, 'data/processed/')
for f in glob.glob('/kaggle/input/**/bpe_train_v3b.bin', recursive=True):
    shutil.copy2(f, 'data/processed/')
for f in glob.glob('/kaggle/input/**/bpe_val_v3b.bin', recursive=True):
    shutil.copy2(f, 'data/processed/')

# Doğrula
!ls -lh data/processed/
```

---

### Hücre 2 — Gerçek 50M Config Oluşturma (1400 Hidden, Beta 0.97, Batch 16)
```python
import yaml, os
os.makedirs('configs', exist_ok=True)
cfg = {
    'data': {
        'block_size': 128,
        'batch_size': 16,        # 1400 hidden VRAM koruması için 16
        'train_bin': 'data/processed/bpe_train_v3b.bin',
        'val_bin': 'data/processed/bpe_val_v3b.bin'
    },
    'model': {
        'vocab_size': 16384,
        'embed_dim': 1400,       # Gerçek 50M mimarisi (49.81M parametre)
        'hidden_size': 1400,
        'num_steps': 32,
        'beta': 0.97,            # Spike oranını %50 civarında tutar
        'threshold': 1.0,
        'tie_weights': False,
        'use_cython': False
    },
    'training': {
        'epochs': 1,
        'learning_rate': 0.0003,
        'optimizer': 'adam',
        'warmup_steps': 500,
        'clip_grad_norm': 1.0,
        'checkpoint_every': 2000,
        'resume': 'auto',
        'checkpoint_dir': 'checkpoints',
        'device': 'cuda'
    }
}
with open('configs/stage6_50m_16k.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print("Gerçek 50M Config hazır: configs/stage6_50m_16k.yaml")
```

---

### Hücre 3 — Tam Epoch Eğitimi Başlatma (Blocking, ~10-12 saat, max_steps_per_epoch=None)
```python
!mkdir -p runs
# max_steps_per_epoch=None (tam 1 epoch / 2B token)
!python scripts/train_lm.py --config configs/stage6_50m_16k.yaml 2>&1 | tee runs/stage6_50m_train.log
```

---

### Hücre 4 — Checkpoint'leri Output'a Kopyalama ve İndirme Linkleri
```python
import os, shutil, glob
from IPython.display import FileLink, display

os.makedirs('/kaggle/working/output_checkpoints', exist_ok=True)
checkpoints = glob.glob('checkpoints/stage6_50m_*.pt')
for f in checkpoints:
    dst = os.path.join('/kaggle/working/output_checkpoints', os.path.basename(f))
    shutil.copy2(f, dst)
    print(f"✔ Kopyalandı: {dst} ({os.path.getsize(dst)/(1024*1024):.1f} MB)")

print("\nTek tek indirme linkleri:")
for f in glob.glob('/kaggle/working/output_checkpoints/*.pt'):
    display(FileLink(os.path.relpath(f, '/kaggle/working')))
```

---

### Hücre 5 — Resume (Oturum koptuğunda son checkpoint'ten devam)
```python
# Oturum kesintisi sonrası en son checkpoint'ten tam senkronize devam:
!python scripts/train_lm.py --config configs/stage6_50m_16k.yaml --resume auto 2>&1 | tee -a runs/stage6_50m_train.log
```

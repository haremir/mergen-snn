# Kaggle Stage 6 (Gerçek 50M SNN — v3 936M Token Tek Oturum) Eğitim Rehberi

Bu rehber, **49.8M parametreli** gerçek Mergen-SNN modelinin dengeli v3 korpusu (**936M token**, tek Kaggle GPU oturumunda tamamlanabilir) ile eğitilmesi için gerekli 6 hücreyi içerir.

---

### Hücre 1 — Kurulum ve Veri Hazırlığı (Dirençli Arama & Hızlı Sembolik Bağlantı)
```python
import os, shutil, glob
from pathlib import Path

# Repo klonlama ve güncelleme
if not os.path.exists('/kaggle/working/mergen-snn'):
    !git clone https://github.com/haremir/mergen-snn.git /kaggle/working/mergen-snn
%cd /kaggle/working/mergen-snn
!git pull

# Bağımlılıklar (Kaggle PyTorch ortamı korunur)
!pip install -q snntorch pyyaml rich numpy datasets tokenizers
!pip install -q -e . --no-deps

# Dizinleri hazırla
os.makedirs('data/processed', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# /kaggle/input taraması
print("--- /kaggle/input Taraması ---")
train_files = glob.glob('/kaggle/input/**/bpe_train*.bin', recursive=True)
val_files = glob.glob('/kaggle/input/**/bpe_val*.bin', recursive=True)
tok_files = glob.glob('/kaggle/input/**/tokenizer*.json', recursive=True)

print(f"Bulunan train dosyaları: {train_files}")
print(f"Bulunan val dosyaları: {val_files}")
print(f"Bulunan tokenizer dosyaları: {tok_files}")

if not train_files:
    print("\n❌ DİKKAT: /kaggle/input altında hiçbir bpe_train*.bin bulunamadı!")
    print("👉 Kaggle notebook sağ panelindeki '+ Add Input' butonuna tıklayarak veri setini ekleyin.")
    print("Mevcut /kaggle/input içeriği:")
    !ls -la /kaggle/input/
else:
    # Train dosyasını bağla / kopyala
    v3_train = [f for f in train_files if 'v3' in f]
    chosen_train = v3_train[0] if v3_train else train_files[0]
    dst_train = 'data/processed/bpe_train_v3.bin'
    if not os.path.exists(dst_train):
        try:
            os.symlink(chosen_train, dst_train)
            print(f"✔ Sembolik bağlandı (hızlı & sıfır disk alanı): {chosen_train} -> {dst_train}")
        except Exception:
            shutil.copy2(chosen_train, dst_train)
            print(f"✔ Kopyalandı: {chosen_train} -> {dst_train}")

    # Val dosyasını bağla / kopyala
    if val_files:
        v3_val = [f for f in val_files if 'v3' in f]
        chosen_val = v3_val[0] if v3_val else val_files[0]
        dst_val = 'data/processed/bpe_val_v3.bin'
        if not os.path.exists(dst_val):
            try:
                os.symlink(chosen_val, dst_val)
                print(f"✔ Sembolik bağlandı: {chosen_val} -> {dst_val}")
            except Exception:
                shutil.copy2(chosen_val, dst_val)
                print(f"✔ Kopyalandı: {chosen_val} -> {dst_val}")

    # Tokenizer dosyasını kopyala
    if tok_files:
        v2_tok = [f for f in tok_files if 'v2' in f]
        chosen_tok = v2_tok[0] if v2_tok else tok_files[0]
        dst_tok = 'data/processed/tokenizer_bpe_v2.json'
        if not os.path.exists(dst_tok):
            shutil.copy2(chosen_tok, dst_tok)
            print(f"✔ Kopyalandı: {chosen_tok} -> {dst_tok}")

# Doğrula
print("\n--- data/processed/ Durumu ---")
!ls -lh data/processed/
```

---

### Hücre 2 — Config Oluşturma ve Zorunlu Doğrulama (Threshold 0.5, Spike Reg 1.0)
```python
config_yaml = """
data:
  block_size: 128
  batch_size: 16
  train_bin: data/processed/bpe_train_v3.bin
  val_bin: data/processed/bpe_val_v3.bin

model:
  vocab_size: 16384
  embed_dim: 1400
  hidden_size: 1400
  num_steps: 32
  beta: 0.97
  threshold: 0.5
  tie_weights: false
  use_cython: false

training:
  epochs: 1
  learning_rate: 0.0003
  optimizer: adam
  warmup_steps: 500
  clip_grad_norm: 1.0
  checkpoint_every: 2000
  resume: auto
  checkpoint_dir: checkpoints
  device: cuda
  spike_reg_weight: 1.0
  target_spike: 0.25
"""
with open('configs/stage6_50m_16k.yaml', 'w') as f:
    f.write(config_yaml.strip() + "\n")

# Zorunlu doğrulama (threshold 0.5 değilse hata fırlat)
import yaml
cfg = yaml.safe_load(open('configs/stage6_50m_16k.yaml'))
assert cfg['model']['threshold'] == 0.5, f"threshold {cfg['model']['threshold']} - 0.5 olmalı!"
assert cfg['training']['spike_reg_weight'] == 1.0
assert cfg['training']['target_spike'] == 0.25
print("✅ Config doğru: threshold=0.5, spike_reg=1.0, target_spike=0.25")
!cat configs/stage6_50m_16k.yaml | grep -E "threshold|beta|spike_reg|target_spike"
```

---

### Hücre 3 — Eski Checkpoint'leri Temizleme (Sıfırdan Başlatma)
```python
!rm -f checkpoints/stage6_50m_*.pt
!rm -f runs/stage6_50m_train.log
!ls checkpoints/
print("✅ Eski checkpoint'ler silindi, sıfırdan başlayacak")
```

---

### Hücre 4 — Eğitimi Başlatma (Tek Oturum, 1 Tam Epoch)
```python
!mkdir -p runs
!python scripts/train_lm.py --config configs/stage6_50m_16k.yaml 2>&1 | tee runs/stage6_50m_train.log
```

---

### Hücre 5 — Checkpoint'leri Output'a Kopyalama
```python
import shutil, glob, os
os.makedirs('/kaggle/working/output_checkpoints', exist_ok=True)
for f in glob.glob('checkpoints/stage6_50m_*.pt'):
    shutil.copy2(f, '/kaggle/working/output_checkpoints/')
    print(f"Kopyalandı: {f}")
```

---

### Hücre 6 — Resume (Kaldığı Yerden Devam Etme)
```python
!python scripts/train_lm.py --config configs/stage6_50m_16k.yaml --resume auto 2>&1 | tee -a runs/stage6_50m_train.log
```

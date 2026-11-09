# Kaggle Commit Mode ile 50M BPE SNN Eğitimi Rehberi

Bu rehber, **49.8M parametreli Mergen-SNN** modelinin Kaggle üzerinde **Commit Mode (Save & Run All)** kullanılarak 9 saatlik kesintisiz GPU oturumunda nasıl eğitileceğini ve çıktıların nasıl indirileceğini adım adım açıklar.

---

## Neden İnteraktif Mod Değil de "Commit Mode"?

| Özellik | İnteraktif Oturum (Interactive Session) | Commit Modu (Save & Run All) |
| :--- | :--- | :--- |
| **Bağlantı Güvenilirliği** | Tarayıcı kapandığında veya internet koptuğunda oturum kısa süre sonra ölür. | **Tamamen bağımsız pod:** Tarayıcıyı veya bilgisayarı kapatabilirsiniz, eğitim arka planda devam eder. |
| **Zaman Sınırı** | Boşta kalma (idle timeout: 20-60 dk) riski yüksektir. | **Kesintisiz 9 saat:** Maksimum GPU süresini tam olarak kullanır. |
| **Çıktı & Model İndirme** | Oturum düştüğünde disk silinir, modeller kaybolabilir. | Oturum bittiğinde tüm `/kaggle/working/` dosyaları **Output** sekmesine kalıcı yazılır; tek tıkla indirilir. |
| **Yeniden Üretilebilirlik** | Geçmiş adımlar hücre sırasına göre karışabilir. | 1'den 6'ya sırayla deterministik koşar, tam log kaydı üretir. |

---

## 1. Hazırlık ve Kurulum Adımları

1. **Notebook Dosyasını İndirin veya Açın:**
   - Projenizdeki `notebooks/stage6_50m_commit.ipynb` dosyasını Kaggle'da yeni bir notebook olarak açın (**File -> Upload Notebook**).

2. **GPU Hızlandırıcıyı Açın:**
   - Sağ paneldeki **Settings** sekmesinden **Accelerator** seçeneğini `GPU P100` veya `GPU T4 x2` olarak belirleyin.
   - **Internet** seçeneğinin `Internet on` (Açık) olduğundan emin olun (GitHub reposunu ve kütüphaneleri çekmek için).

3. **Veri Setini Ekleyin (+ Add Input):**
   - Sağ üstteki **+ Add Input** butonuna tıklayın.
   - Mergen-SNN veri setinizi (`bpe_train_v3.bin`, `bpe_val_v3.bin`, `tokenizer_bpe_v2.json` içeren veri kümesi) aratıp ekleyin.

---

## 2. Notebook Hücreleri ve Görevleri

Notebook toplam **6 hücreden** oluşur:

### Hücre 1: Kurulum ve Veri Hazırlığı
- GitHub reposunu klonlar veya son değişiklikleri (`git pull`) çeker.
- `pip install -e .` ile paketi kurar.
- `/kaggle/input` altındaki eğitim ve doğrulama dosyalarını tespit edip `data/processed/` dizinine sembolik bağlar (sıfır disk tüketimi, maksimum hız).

```python
# Hücre 1: Kurulum ve Repo Hazırlığı
import os, sys, shutil, glob

if not os.path.exists('/kaggle/working/mergen-snn'):
    !git clone https://github.com/haremir/mergen-snn.git /kaggle/working/mergen-snn

%cd /kaggle/working/mergen-snn
!git pull

# Bağımlılıkları kur
!pip install -q snntorch pyyaml rich numpy datasets tokenizers
!pip install -q -e . --no-deps

os.makedirs('data/processed', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Veri setini bağla (/kaggle/input altından otomatik bulma ve sembolik bağlama)
train_files = glob.glob('/kaggle/input/**/bpe_train*.bin', recursive=True)
val_files = glob.glob('/kaggle/input/**/bpe_val*.bin', recursive=True)
tok_files = glob.glob('/kaggle/input/**/tokenizer*.json', recursive=True)

print(f'Bulunan train dosyaları: {train_files}')
print(f'Bulunan val dosyaları: {val_files}')
print(f'Bulunan tokenizer dosyaları: {tok_files}')

if train_files:
    v3_train = [f for f in train_files if 'v3' in f]
    chosen_train = v3_train[0] if v3_train else train_files[0]
    dst_train = 'data/processed/bpe_train_v3.bin'
    if not os.path.exists(dst_train):
        try:
            os.symlink(chosen_train, dst_train)
            print(f'[OK] Sembolik bağlandı: {chosen_train} -> {dst_train}')
        except Exception:
            shutil.copy2(chosen_train, dst_train)
            print(f'[OK] Kopyalandı: {chosen_train} -> {dst_train}')

if val_files:
    v3_val = [f for f in val_files if 'v3' in f]
    chosen_val = v3_val[0] if v3_val else val_files[0]
    dst_val = 'data/processed/bpe_val_v3.bin'
    if not os.path.exists(dst_val):
        try:
            os.symlink(chosen_val, dst_val)
            print(f'[OK] Sembolik bağlandı: {chosen_val} -> {dst_val}')
        except Exception:
            shutil.copy2(chosen_val, dst_val)
            print(f'[OK] Kopyalandı: {chosen_val} -> {dst_val}')

if tok_files:
    chosen_tok = tok_files[0]
    dst_tok = 'data/processed/tokenizer_bpe_v2.json'
    if not os.path.exists(dst_tok):
        try:
            os.symlink(chosen_tok, dst_tok)
        except Exception:
            shutil.copy2(chosen_tok, dst_tok)
    print(f'[OK] Tokenizer hazır: {dst_tok}')
```

---

### Hücre 2: Veri Doğrulama ve Boyut Kontrolü
- Gerekli tüm binary veri kümelerinin ve tokenizer dosyasının mevcut olduğunu ve 0 byte olmadığını garanti eder.

```python
# Hücre 2: Veri Doğrulama ve Boyut Kontrolü
import os

req_files = [
    'data/processed/bpe_train_v3.bin',
    'data/processed/bpe_val_v3.bin',
    'data/processed/tokenizer_bpe_v2.json'
]

for rf in req_files:
    assert os.path.exists(rf), f'HATA: Gerekli dosya eksik: {rf}'
    size = os.path.getsize(rf)
    assert size > 0, f'HATA: Dosya boş (0 byte): {rf}'
    print(f'[OK] {rf:<38}: {size / (1024*1024):.2f} MB')

print('\n[OK] Tüm veri dosyaları mevcut ve doğrulanmıştır.')
```

---

### Hücre 3: Config Dosyasını Hardcoded Yazma ve Assert Doğrulaması
- Eğitim konfigürasyonunu doğrudan YAML formatında diske yazar ve hemen ardından `assert` ile parametreleri doğrular.
- **Kritik Hiper-parametreler:**
  - `threshold: 0.5` (Düşük eşik ile nöron aktivasyonu korunur)
  - `beta: 0.97` (Uzun süreli temporal bellek)
  - `spike_reg_weight: 10.0` (Spike oranını hedefte tutan ceza katsayısı)
  - `target_spike: 0.15` (Hedef firing rate %15-30)
  - `max_steps_per_epoch: 380000` (9 saat sınırına tam uyumlu adım sayısı)

```python
# Hücre 3: Config Dosyasını Hardcoded Yazma ve Assert Kontrolleri
import yaml

config_yaml = """data:
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
  spike_reg_weight: 10.0
  target_spike: 0.15
  max_steps_per_epoch: 380000
"""

config_path = 'configs/stage6_50m_16k.yaml'
with open(config_path, 'w', encoding='utf-8') as f:
    f.write(config_yaml.strip() + '\n')

# Assert Kontrolleri
with open(config_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

assert cfg['model']['threshold'] == 0.5, f"Threshold hatası: {cfg['model']['threshold']}"
assert cfg['model']['beta'] == 0.97, f"Beta hatası: {cfg['model']['beta']}"
assert cfg['training']['spike_reg_weight'] == 10.0, f"Spike reg hatası: {cfg['training']['spike_reg_weight']}"
assert cfg['training']['target_spike'] == 0.15, f"Target spike hatası: {cfg['training']['target_spike']}"
assert cfg['training']['max_steps_per_epoch'] == 380000, f"Max steps hatası: {cfg['training']['max_steps_per_epoch']}"

print('[OK] Config başarıyla yazıldı ve tüm assert testlerinden geçti.')
print(f"  threshold: {cfg['model']['threshold']}")
print(f"  beta: {cfg['model']['beta']}")
print(f"  spike_reg_weight: {cfg['training']['spike_reg_weight']}")
print(f"  target_spike: {cfg['training']['target_spike']}")
print(f"  max_steps_per_epoch: {cfg['training']['max_steps_per_epoch']}")
```

---

### Hücre 4: Eğitimi Başlatma
- Modeli GPU üzerinde eğitir, her 100 adımda bir `loss`, `ppl`, `bpb` ve `spk2` oranlarını raporlar.

```python
# Hücre 4: 50M BPE SNN Model Eğitimini Başlat
!python scripts/train_lm.py --config configs/stage6_50m_16k.yaml
```

---

### Hücre 5: Checkpointleri Kaggle Output Kök Dizinine Aktarma
- Kaggle ortamında `/kaggle/working/` kök dizinindeki tüm dosyalar çalıştırma bittiğinde kalıcı **Output Artifacts** haline gelir.
- Bu hücre, `checkpoints/` altındaki model ağırlıklarını (`.pt`) ve eğitim eğrisi grafiğini (`.png`) `/kaggle/working/` altına kopyalayarak arayüzden tek tıkla indirilebilir hale getirir.

```python
# Hücre 5: Checkpointleri ve Çıktıları Kaggle Output Kök Dizinine Kopyala
import os, shutil, glob

out_root = '/kaggle/working'
ckpt_dir = '/kaggle/working/mergen-snn/checkpoints'

print('--- Checkpoint Kopyalama ---')
pt_files = glob.glob(os.path.join(ckpt_dir, '*.pt'))
png_files = glob.glob(os.path.join(ckpt_dir, '*.png'))

for src_f in pt_files + png_files:
    fname = os.path.basename(src_f)
    dst_f = os.path.join(out_root, fname)
    shutil.copy2(src_f, dst_f)
    sz_mb = os.path.getsize(dst_f) / (1024 * 1024)
    print(f'[OK] Kopyalandı -> {dst_f} ({sz_mb:.2f} MB)')

print('\n--- /kaggle/working/ Dosya Listesi (İndirilebilir Çıktılar) ---')
!ls -lh /kaggle/working/*.pt /kaggle/working/*.png
```

---

### Hücre 6: İnference ve Üretim Doğrulama Testi
- Kaydedilen en iyi modeli (`stage6_50m_best.pt` veya `stage6_50m_final.pt`) yükler.
- CS, Türkçe ve Python promptları ile örnek üretimler gerçekleştirir; perplexity (`ppl`) ve spike firing rate (`spk2`) metriklerini yazdırır.

```python
# Hücre 6: İnference ve Doğrulama Testi
import os, torch
from tokenizers import Tokenizer
from mergen.models.snn_lm_bpe import BpeSNNLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_candidates = [
    '/kaggle/working/stage6_50m_best.pt',
    '/kaggle/working/stage6_50m_final.pt',
    'checkpoints/stage6_50m_best.pt',
    'checkpoints/stage6_50m_final.pt',
]
ckpt_path = next((p for p in ckpt_candidates if os.path.exists(p)), None)
assert ckpt_path is not None, 'HATA: İnference için checkpoint bulunamadı!'

print(f'Model yükleniyor: {ckpt_path}')
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt.get('model_state_dict', ckpt)

model = BpeSNNLM(vocab_size=16384, embed_dim=1400, hidden_size=1400, beta=0.97, threshold=0.5, use_cython=False, tie_weights=False)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

tok_path = 'data/processed/tokenizer_bpe_v2.json'
tok = Tokenizer.from_file(tok_path)

prompts = [
    'Bilgisayar bilimlerinde yapay zeka',
    'def quicksort(arr):',
    'Spiking neural networks are'
]

print('=' * 60)
print('Mergen-SNN 50M Üretim Doğrulama Testi')
print('=' * 60)

for prompt in prompts:
    enc = tok.encode(prompt)
    gen_ids, metrics = model.generate_with_metrics(enc.ids, max_new_tokens=40, temperature=0.7)
    gen_text = tok.decode(enc.ids + gen_ids)
    print(f'\nPrompt: {prompt}')
    print(f'Üretim: {gen_text}')
    print(f"Metrikler: spk2={metrics['spike_ratio']:.3f} | loss={metrics['loss']:.3f} | ppl={metrics['ppl']:.2f} | bpb={metrics['bpb']:.3f}")
print('=' * 60)
```

---

## 3. Çalıştırma Talimatı (Commit Modu)

1. Sayfanın sağ üst köşesindeki **Save Version** mavi butonuna tıklayın.
2. Açılan pencerede **Version Type** olarak:
   - **`Save & Run All (Commit)`** seçeneğini işaretleyin.
3. **Save** butonuna basarak işi gönderin.
4. Artık tarayıcıyı veya bilgisayarınızı güvenle kapatabilirsiniz.
5. İş tamamlandığında (veya sol alttan versiyon loglarını incelerken) Kaggle notebook ana sayfasındaki **Output** sekmesine gidin.
6. `stage6_50m_best.pt`, `stage6_50m_final.pt` ve `stage6_50m_training_curve.png` dosyalarının yanında bulunan **Download** butonuna tıklayarak modellerinizi bilgisayarınıza indirin!

"""Mergen-SNN — Colab 50M Canlı İzleme ve Keep-Alive Daemon Scripti.

1. Colab oturumuna her 45 saniyede bir heartbeat göndererek Colab'ın
   inactivity/idle reaper tarafından kapatılmasını önler.
2. Veri hazırlığı (data prep) ve model eğitimi (train_lm.py) loglarını canlı okur.
3. GPU kullanımını, spk1/spk2 oranlarını ve ölü nöron yüzdelerini ekrana basar.
4. Yeni checkpoint (.pt) üretildikçe yerel 'checkpoints' klasörüne otomatik indirir.
"""

import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CKPT_DIR = PROJECT_ROOT / "checkpoints"
LOCAL_CKPT_DIR.mkdir(parents=True, exist_ok=True)

PROBE_CODE = r'''
import os, glob, subprocess, json

pipe_log = "/content/mergen-snn/runs/pipeline.log"
data_log = "/content/mergen-snn/runs/data_prep.log"
train_log = "/content/mergen-snn/runs/stage6_50m.log"

pipe_tail = ""
if os.path.exists(pipe_log):
    with open(pipe_log, "r", encoding="utf-8", errors="replace") as f:
        pipe_tail = "".join(f.readlines()[-4:]).strip()

data_tail = ""
if os.path.exists(data_log):
    with open(data_log, "r", encoding="utf-8", errors="replace") as f:
        data_tail = "".join(f.readlines()[-4:]).strip()

train_tail = ""
if os.path.exists(train_log):
    with open(train_log, "r", encoding="utf-8", errors="replace") as f:
        train_tail = "".join(f.readlines()[-4:]).strip()

gpu_info = ""
try:
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"]).decode()
    gpu_info = smi.strip()
except Exception:
    gpu_info = "GPU sorgulanamadi"

ps_list = []
try:
    ps = subprocess.check_output(["pgrep", "-fa", "python"]).decode().strip()
    for l in ps.splitlines():
        if any(k in l for k in ["pipeline", "prepare", "train_lm"]):
            ps_list.append(l)
except Exception:
    pass

train_bin_sz = 0
if os.path.exists("/content/mergen-snn/data/processed/bpe_train_v3.bin"):
    train_bin_sz = os.path.getsize("/content/mergen-snn/data/processed/bpe_train_v3.bin") / (1024 * 1024)

ckpts = []
for cd in ["/content/drive/MyDrive/mergen-snn/checkpoints", "/content/mergen-snn/checkpoints"]:
    if os.path.exists(cd):
        for f in glob.glob(f"{cd}/*.pt"):
            ckpts.append({"name": os.path.basename(f), "size_mb": os.path.getsize(f) / (1024 * 1024), "path": f})

print("---JSON_OUTPUT---")
print(json.dumps({
    "pipe": pipe_tail,
    "data": data_tail,
    "train": train_tail,
    "gpu": gpu_info,
    "procs": ps_list,
    "train_mb": round(train_bin_sz, 2),
    "ckpts": ckpts
}))
'''

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "mergen-50m"
    log(f"Colab Keep-Alive & Metric Daemon başlatıldı (Oturum: {session})...")

    known_ckpts = set()
    consecutive_errs = 0

    while True:
        try:
            cmd = ["uv", "tool", "run", "colabapi", "raw", "--", "exec", "-s", session, "--timeout", "30"]
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            res = subprocess.run(
                cmd,
                input=PROBE_CODE,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=45
            )

            if res.returncode == 0 and "---JSON_OUTPUT---" in res.stdout:
                consecutive_errs = 0
                parts = res.stdout.split("---JSON_OUTPUT---")
                import json
                info = json.loads(parts[1].strip())

                # Ekran çıktısı oluştur
                gpu_str = info.get("gpu", "")
                train_mb = info.get("train_mb", 0)
                procs = info.get("procs", [])
                proc_str = " | ".join(p.split()[-1] if len(p.split()) > 1 else p for p in procs) if procs else "Yok"

                train_log = info.get("train", "")
                data_log = info.get("data", "")

                if train_log:
                    # Model eğitimi aşamasındayız
                    last_line = train_log.splitlines()[-1] if train_log.splitlines() else ""
                    log(f"🔥 [EGITIM] GPU: {gpu_str} | {last_line}")
                elif data_log:
                    # Veri hazırlığı aşamasındayız
                    last_line = data_log.splitlines()[-1] if data_log.splitlines() else ""
                    log(f"📦 [VERI URETIMI] Train: {train_mb:.1f} MB | {last_line}")
                else:
                    log(f"⏳ [BEKLEMEDE] GPU: {gpu_str} | Aktif Surecler: {proc_str}")

                # Checkpoint kontrolü & indirme
                for ckpt in info.get("ckpts", []):
                    cname = ckpt["name"]
                    if cname not in known_ckpts:
                        known_ckpts.add(cname)
                        log(f"⭐ YENI CHECKPOINT TESPIT EDILDI: {cname} ({ckpt['size_mb']:.1f} MB)")
                        # Yerel diske indir
                        try:
                            remote_path = ckpt["path"].lstrip("/")
                            down_cmd = ["uv", "tool", "run", "colabapi", "raw", "--", "download", "-s", session, remote_path, str(LOCAL_CKPT_DIR / cname)]
                            subprocess.run(down_cmd, capture_output=True, timeout=120)
                            log(f"✔ Checkpoint yerel klasöre yedeklendi: checkpoints/{cname}")
                        except Exception as e:
                            log(f"Checkpoint indirme hatası: {e}")
            else:
                consecutive_errs += 1
                log(f"⚠️ Keep-alive uyarısı ({consecutive_errs}): returncode={res.returncode}")
                if consecutive_errs >= 6:
                    log("❌ Oturum düşmüş olabilir veya yanıt vermiyor. Kontrol ediliyor...")

        except subprocess.TimeoutExpired:
            log("⚠️ Keep-alive zaman aşımına uğradı (45s), bir sonraki döngüde yeniden denenecek.")
        except Exception as e:
            log(f"Daemon döngü hatası: {e}")

        time.sleep(45)

if __name__ == "__main__":
    main()

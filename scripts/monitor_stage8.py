"""Mergen-SNN — Stage 8 100M Hibrit Model Colab Monitor Daemon.

1. Rapor Sıklığı: 30-45 dakikada bir (varsayılan: 1800 saniye / 30 dk).
2. Manuel Rapor Trigger: 'runs/report_now.trigger' veya 'python scripts/monitor_stage8.py --trigger'
   ile anlık durum raporu üretir.
3. Checkpoint Senkronizasyonu: Yeni weights.pt veya best.pt dosyalarını yerel checkpoints'e indirir.
4. Metrik Takibi: Loss, PPL, BPB, Spike Rate, Step/s, ETA ve GPU takibi yapar.
"""

import argparse
import glob
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Dict, Any, Optional

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CKPT_DIR = PROJECT_ROOT / "checkpoints"
LOCAL_CKPT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = PROJECT_ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_JSON = RUNS_DIR / "stage8_progress.json"
MONITOR_LOG = RUNS_DIR / "stage8_monitor.log"
TRIGGER_FILE = RUNS_DIR / "report_now.trigger"

DEFAULT_SESSION = "mergen-spike-10m"
DEFAULT_REPORT_INTERVAL = 1800  # 30 dakika (1800 saniye)

PROBE_CODE = r'''
import os, glob, subprocess, json

train_log = "/content/mergen-snn/runs/stage8_hybrid_100m.log"
train_tail = ""
if os.path.exists(train_log):
    with open(train_log, "r", encoding="utf-8", errors="replace") as f:
        train_tail = "".join(f.readlines()[-12:]).strip()

gpu_info = ""
try:
    smi = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits"
    ]).decode().strip()
    gpu_info = smi
except Exception:
    gpu_info = "GPU sorgulanamadi"

ps_list = []
try:
    ps = subprocess.check_output(["pgrep", "-fa", "python"]).decode().strip()
    for l in ps.splitlines():
        if "train_lm.py" in l:
            ps_list.append(l)
except Exception:
    pass

ckpts = []
cd = "/content/mergen-snn/checkpoints"
if os.path.exists(cd):
    for f in glob.glob(f"{cd}/stage8*.pt"):
        ckpts.append({
            "name": os.path.basename(f),
            "size_mb": round(os.path.getsize(f) / (1024 * 1024), 2),
            "path": f
        })

print("---JSON_OUTPUT---")
print(json.dumps({
    "train_tail": train_tail,
    "gpu": gpu_info,
    "train_running": len(ps_list) > 0,
    "procs": ps_list,
    "ckpts": ckpts
}))
'''


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(MONITOR_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def cleanup_stale_locks():
    for f in glob.glob(os.path.expanduser("~/.config/colab-cli/*.lock")):
        try:
            os.remove(f)
        except OSError:
            pass


def cleanup_local_old_steps():
    """Yerel disk dolmasın diye stage8 eski step checkpoint'lerini temizle, son 2'yi tut."""
    step_files = sorted(
        [f for f in LOCAL_CKPT_DIR.glob("stage8_hybrid_100m_step*") if not f.name.endswith(".png")],
        key=lambda p: p.stat().st_mtime
    )
    if len(step_files) > 2:
        for old_f in step_files[:-2]:
            try:
                old_f.unlink()
                log(f"🧹 Yerel disk temizliği: Eski step silindi: {old_f.name}")
            except OSError:
                pass


def sync_download(session: str, remote_path: str, local_path: Path, cname: str):
    try:
        log(f"📥 Checkpoint senkronize ediliyor: {cname}")
        cmd = ["uv", "tool", "run", "colabapi", "raw", "--", "download", "-s", session, remote_path, str(local_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if res.returncode == 0:
            log(f"✅ Checkpoint başarıyla indirildi: {cname}")
            cleanup_local_old_steps()
        else:
            log(f"⚠️ Checkpoint indirme hatası ({cname}): {res.stderr.strip() or res.stdout.strip()}")
    except Exception as e:
        log(f"⚠️ Checkpoint indirme istisnası ({cname}): {e}")


def run_colab_probe(session: str, timeout: int = 40):
    cleanup_stale_locks()
    cmd = ["uv", "tool", "run", "colabapi", "raw", "--", "exec", "-s", session, "--timeout", "30"]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        p = subprocess.run(
            cmd,
            input=PROBE_CODE,
            text=True,
            capture_output=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=env
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        cleanup_stale_locks()
        return -1, "", "Timeout"
    except Exception as e:
        return -1, "", str(e)


def parse_metrics_from_tail(log_tail: str) -> Dict[str, Any]:
    """Log kuyruğundan son adım, loss, ppl, spk_rate ve hız bilgilerini çeker."""
    metrics = {
        "step": None,
        "total_steps": None,
        "loss": None,
        "ppl": None,
        "spk_rate": None,
        "speed": None,
        "lr": None,
        "last_line": "",
    }
    lines = [l.strip() for l in log_tail.splitlines() if l.strip()]
    if not lines:
        return metrics

    metrics["last_line"] = lines[-1]

    # Step satırlarını geriden öne ara
    for line in reversed(lines):
        # Örnek: Step 2400/1953125 (Total 2400) | loss=8.1234 (cum:8.20) | ppl=3380.5 | bpb=3.56 | lr=0.000300 | spk=18.4% | 5.2step/s
        m_step = re.search(r"Step\s+(\d+)/(\d+)", line)
        if m_step:
            metrics["step"] = int(m_step.group(1))
            metrics["total_steps"] = int(m_step.group(2))

        m_loss = re.search(r"loss=([\d\.]+)", line)
        if m_loss and metrics["loss"] is None:
            metrics["loss"] = float(m_loss.group(1))

        m_ppl = re.search(r"ppl=([\d\.]+)", line)
        if m_ppl and metrics["ppl"] is None:
            metrics["ppl"] = float(m_ppl.group(1))

        m_spk = re.search(r"spk=([\d\.]+)%", line)
        if m_spk and metrics["spk_rate"] is None:
            metrics["spk_rate"] = float(m_spk.group(1))

        m_spd = re.search(r"([\d\.]+)step/s", line)
        if m_spd and metrics["speed"] is None:
            metrics["speed"] = float(m_spd.group(1))

        m_lr = re.search(r"lr=([\d\.]+)", line)
        if m_lr and metrics["lr"] is None:
            metrics["lr"] = float(m_lr.group(1))

        if metrics["step"] is not None:
            break

    return metrics


def print_formal_report(data: Dict[str, Any], trigger_type: str = "PERİYODİK (30 DK)"):
    """Kullanıcıya veya loga düzenli metrik karşılaştırma raporu basar."""
    m = data.get("metrics", {})
    gpu = data.get("gpu", "N/A")
    running = "🟢 AKTİF ÇALIŞIYOR" if data.get("train_running") else "⏸️ DURDU / BEKLEMEDE"
    step = m.get("step")
    tot = m.get("total_steps") or 1953125
    pct = (step / tot * 100) if step and tot else 0.0

    speed = m.get("speed")
    rem_steps = (tot - step) if step and tot else None
    eta_str = f"{(rem_steps / max(speed, 0.1)) / 3600:.1f} saat" if (rem_steps and speed) else "Hesaplanıyor"

    sep = "=" * 65
    print("\n" + sep)
    print(f"📊 [STAGE 8 - 100M HİBRİT RAPORU] — {trigger_type}")
    print(f"🕒 Zaman: {time.strftime('%Y-%m-%d %H:%M:%S')} | Durum: {running}")
    print(f"🎮 GPU (T4): {gpu}")
    print(sep)
    if step is not None:
        print(f"📈 İlerleme     : Adım {step:,} / {tot:,} (%{pct:.2f})")
        print(f"📉 Train Loss   : {m.get('loss', 'N/A')} | PPL: {m.get('ppl', 'N/A')}")
        print(f"⚡ Spike Rate   : %{m.get('spk_rate', 'N/A')} (Hedef: %15-30)")
        print(f"🚀 Hız / ETA    : {speed or 'N/A'} step/s | Kalan: {eta_str}")
        print(f"🎯 Öğrenme Oranı: {m.get('lr', 'N/A')}")
    else:
        print(f"📄 Son Log Satırı: {m.get('last_line', 'Henüz log satırı yok')}")

    ckpts = data.get("checkpoints", [])
    if ckpts:
        print(f"💾 Mevcut Checkpoint'ler ({len(ckpts)} adet):")
        for c in ckpts[-3:]:
            print(f"   • {c['name']} ({c['size_mb']} MB)")
    print(sep + "\n")


def check_and_clear_trigger() -> bool:
    """Manuel rapor tetikleme dosyasını kontrol eder."""
    triggered = False
    for path in [TRIGGER_FILE, Path("/tmp/report_now")]:
        if path.exists():
            try:
                path.unlink()
                triggered = True
            except OSError:
                pass
    return triggered


def main():
    parser = argparse.ArgumentParser(description="Stage 8 Colab Monitor Daemon")
    parser.add_argument("--session", type=str, default=DEFAULT_SESSION, help="Colab oturum adı")
    parser.add_argument("--interval", type=int, default=DEFAULT_REPORT_INTERVAL, help="Rapor sıklığı (saniye, örn 1800=30dk, 2700=45dk)")
    parser.add_argument("--trigger", action="store_true", help="Monitor'a anlık rapor üretme emri ver (trigger dosyası oluşturur)")
    parser.add_argument("--status", action="store_true", help="Son kaydedilen ilerlemeyi oku ve ekrana bas")
    args = parser.parse_args()

    # CLI Komutu: Trigger oluştur
    if args.trigger:
        TRIGGER_FILE.write_text("trigger\n", encoding="utf-8")
        print(f"✅ Anlık rapor tetikleme dosyası oluşturuldu: {TRIGGER_FILE}")
        return

    # CLI Komutu: Anlık durumu oku
    if args.status:
        if PROGRESS_JSON.exists():
            with open(PROGRESS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            print_formal_report(data, trigger_type="ANLIK KULLANICI SORGUSU")
        else:
            print("Henüz kaydedilmiş durum dosyası bulunamadı.")
        return

    log(f"🚀 Stage 8 Monitor Daemon başlatıldı (Oturum: {args.session}, Rapor Aralığı: {args.interval}s / {args.interval/60:.0f} dk)...")

    last_report_time = 0
    seen_ckpts = {f.name for f in LOCAL_CKPT_DIR.glob("*.pt")}

    while True:
        cleanup_stale_locks()
        now = time.time()
        is_triggered = check_and_clear_trigger()
        time_for_report = (now - last_report_time >= args.interval)

        # Her 60 saniyede bir Colab'ı sorgula
        retcode, stdout, stderr = run_colab_probe(args.session, timeout=40)

        if retcode == 0 and "---JSON_OUTPUT---" in stdout:
            parts = stdout.split("---JSON_OUTPUT---")
            info = json.loads(parts[1].strip())

            train_tail = info.get("train_tail", "")
            metrics = parse_metrics_from_tail(train_tail)

            progress_data = {
                "timestamp": now,
                "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                "gpu": info.get("gpu", ""),
                "train_running": info.get("train_running", False),
                "metrics": metrics,
                "log_tail": train_tail,
                "checkpoints": info.get("ckpts", [])
            }

            with open(PROGRESS_JSON, "w", encoding="utf-8") as f_prog:
                json.dump(progress_data, f_prog, indent=2, ensure_ascii=False)

            # Rapor basma koşulu: Ya zamanı geldi ya da kullanıcı manuel trigger etti
            if time_for_report or is_triggered:
                trigger_label = "MANUEL TRIGGER" if is_triggered else f"PERİYODİK ({args.interval//60} DK)"
                print_formal_report(progress_data, trigger_type=trigger_label)
                last_report_time = now

            # Yeni weights kontrolü ve indirme
            for ckpt in info.get("ckpts", []):
                cname = ckpt["name"]
                if cname not in seen_ckpts:
                    seen_ckpts.add(cname)
                    if "weights.pt" in cname or "best.pt" in cname:
                        log(f"⭐ YENİ MODEL AĞIRLIĞI TESPİT EDİLDİ: {cname} ({ckpt['size_mb']} MB)")
                        sync_download(args.session, ckpt["path"], LOCAL_CKPT_DIR / cname, cname)

        else:
            log(f"⚠️ Colab probe yanıt vermedi (retcode={retcode})")

        time.sleep(15)  # 15 saniyede bir döngü ve trigger kontrolü


if __name__ == "__main__":
    main()

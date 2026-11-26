"""Mergen-SNN — Stage 8 Colab Keep-Alive Daemon.

Colab oturumunu inactivity/idle reaper'dan korumak için 35 saniyede bir
hafif heartbeat (ping) gönderir.
"""

import argparse
import glob
import io
import os
from pathlib import Path
import subprocess
import sys
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "runs" / "keep_alive_stage8.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_SESSION = "mergen-spike-10m"
HEARTBEAT_INTERVAL = 35  # saniye


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def cleanup_stale_locks():
    for f in glob.glob(os.path.expanduser("~/.config/colab-cli/*.lock")):
        try:
            os.remove(f)
        except OSError:
            pass


def send_heartbeat(session: str, timeout: int = 25) -> bool:
    cleanup_stale_locks()
    cmd = [
        "uv", "tool", "run", "colabapi", "raw", "--", "exec",
        "-s", session,
        "--timeout", str(timeout),
    ]
    code = 'import time; print("PONG_" + str(int(time.time())))'
    try:
        p = subprocess.run(
            cmd,
            input=code,
            text=True,
            capture_output=True,
            timeout=timeout + 10,
            encoding="utf-8",
            errors="replace",
        )
        return p.returncode == 0 and "PONG_" in p.stdout
    except Exception as e:
        cleanup_stale_locks()
        return False


def main():
    parser = argparse.ArgumentParser(description="Colab Keep-Alive Daemon")
    parser.add_argument("--session", type=str, default=DEFAULT_SESSION, help="Colab oturum adı")
    parser.add_argument("--interval", type=int, default=HEARTBEAT_INTERVAL, help="Heartbeat aralığı (saniye)")
    args = parser.parse_args()

    log(f"🟢 Stage 8 Keep-Alive Daemon başladı (Oturum: {args.session}, Aralık: {args.interval}s)")

    consecutive_fails = 0
    while True:
        success = send_heartbeat(args.session)
        if success:
            consecutive_fails = 0
            log(f"💓 Heartbeat başarılı (Colab aktif)")
        else:
            consecutive_fails += 1
            log(f"⚠️ Heartbeat başarısız ({consecutive_fails})")
            if consecutive_fails >= 5:
                log("🚨 UYARI: Colab oturumu 5 heartbeat boyunca yanıt vermedi!")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()

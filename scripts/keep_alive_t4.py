import subprocess
import time
import sys

SESSION = "mergen-t4-v3"
print(f"Keep-alive daemon started for session '{SESSION}'...", flush=True)

while True:
    try:
        res = subprocess.run(
            ["uv", "tool", "run", "colabapi", "raw", "--", "exec", "-s", SESSION, "--timeout", "30"],
            input="print('alive')",
            capture_output=True,
            text=True,
            timeout=40
        )
        if res.returncode == 0:
            print(f"[{time.strftime('%H:%M:%S')}] keep-alive OK: {res.stdout.strip()}", flush=True)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] keep-alive warning ({res.returncode}): {res.stderr.strip()}", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] keep-alive hata: {e}", flush=True)
    time.sleep(120)  # 2 dakika


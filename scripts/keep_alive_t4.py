import subprocess
import time
import sys

SESSION = "mergen-t4-v3"
print(f"Keep-alive daemon started for session '{SESSION}'...")

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
            print(f"[{time.strftime('%H:%M:%S')}] keep-alive OK: {res.stdout.strip()}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] keep-alive warning ({res.returncode}): {res.stderr.strip()}")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] keep-alive hata: {e}")
    time.sleep(120)  # 2 dakika

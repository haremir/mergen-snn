import subprocess
import sys
import os

os.chdir("/content/mergen-snn")
res = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], capture_output=True, text=True)
print("Pip install -e . exit code:", res.returncode)
print(res.stdout)
print(res.stderr)

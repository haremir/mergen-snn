import subprocess
import os

os.chdir("/content/mergen-snn")
res = subprocess.run(["git", "pull"], capture_output=True, text=True)
print("Git pull output:")
print(res.stdout)
print(res.stderr)

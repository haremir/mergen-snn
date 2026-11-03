import os
from pathlib import Path
import subprocess

print("=== Colab Environment Check ===")
print("Current Directory:", os.getcwd())
print("Directory listing of /content:")
try:
    print(os.listdir("/content"))
except Exception as e:
    print("Error listing /content:", e)

drive_path = Path("/content/drive")
if drive_path.exists():
    print("/content/drive exists!")
    try:
        print(os.listdir("/content/drive"))
    except Exception as e:
        print("Error listing /content/drive:", e)
else:
    print("/content/drive does NOT exist yet.")

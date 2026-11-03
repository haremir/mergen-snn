import os
import subprocess

print("--- DIRECTORY CHECK ---")
print("cwd:", os.getcwd())
print("contents:", os.listdir("."))

print("--- DRIVE CHECK ---")
print("/content/drive exists?", os.path.exists("/content/drive"))
if os.path.exists("/content/drive"):
    print("drive contents:", os.listdir("/content/drive"))

print("--- DISK USAGE ---")
df = subprocess.run(["df", "-h", "/content"], capture_output=True, text=True)
print(df.stdout)

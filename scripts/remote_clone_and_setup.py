import os
from pathlib import Path
import shutil
import subprocess

repo_dir = Path("/content/mergen-snn")
if repo_dir.exists():
    print("Mevcut mergen-snn temizleniyor...")
    shutil.rmtree(repo_dir)

print(">>> Git clone başlatılıyor...")
subprocess.run(["git", "clone", "https://github.com/haremir/mergen-snn.git", "/content/mergen-snn"], check=True)
print(">>> Git clone tamamlandı!")

print(">>> pip install uv...")
subprocess.run(["pip", "install", "-q", "uv"], check=True)

print(">>> uv sync...")
subprocess.run(["uv", "sync"], cwd=str(repo_dir), check=True)

print(">>> Cython build...")
try:
    subprocess.run(["uv", "run", "python", "src/mergen/cython_ext/setup.py", "build_ext", "--inplace"], cwd=str(repo_dir), check=True)
    print(">>> Cython derleme başarılı!")
except Exception as e:
    print(">>> Cython build uyarısı:", e)

print(">>> pip install snntorch...")
subprocess.run(["pip", "install", "-q", "snntorch"], check=True)

# data/processed ve checkpoints dizinlerini oluştur
(repo_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
(repo_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
print(">>> Ortam hazırlandı!")

import subprocess
import os

subprocess.run(["rm", "-rf", "/content/mergen-snn/src/mergen/data/__pycache__", "/content/mergen-snn/src/mergen/__pycache__"])

with open("/content/mergen-snn/src/mergen/data/char_dataset.py", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines[:35]):
        print(f"{i+1}: {line}", end="")

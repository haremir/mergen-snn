import subprocess
import sys
import os

print("=== PULLING LATEST REPO ON COLAB ===")
os.chdir("/content/mergen-snn")
subprocess.run(["git", "pull"], check=True)

print("=== CHECKING IMPORT ===")
sys.path.insert(0, "/content/mergen-snn/src")
from mergen.models.snn_lm_bpe import BpeSNNLM
from mergen.data.streaming_sources import stream_all_sources
print("Imports verified successfully!")

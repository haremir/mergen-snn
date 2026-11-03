import os
import subprocess
import sys

repo_dir = "/content/mergen-snn"

# 1. Pull latest code
print("Pulling latest code from origin/main...")
subprocess.run(["git", "pull", "origin", "main"], cwd=repo_dir, check=True)

# 2. Test 5 training steps with python3 (CUDA)
env = os.environ.copy()
env["PYTHONPATH"] = f"{repo_dir}/src"

print("Testing 5 training steps with system python on Tesla T4...")
test_cmd = [
    "python3",
    "scripts/train_lm.py",
    "--config", "configs/stage4_small.yaml",
    "--max_steps", "5",
]

res = subprocess.run(test_cmd, cwd=repo_dir, env=env, capture_output=True, text=True)
print("=== Return code:", res.returncode, "===")
print("STDOUT:\n", res.stdout)
if res.stderr:
    print("STDERR:\n", res.stderr)

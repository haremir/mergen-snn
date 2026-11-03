import subprocess

print("Checking nvidia-smi:")
res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(res.stdout)

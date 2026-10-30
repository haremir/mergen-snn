import os
import subprocess
import time
import json
from pathlib import Path

REPO_DIR = Path(r"c:\Users\PC\Desktop\mergen-snn")
LOCAL_CKPT_DIR = REPO_DIR / "checkpoints"
LOCAL_CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_LOG = REPO_DIR / "runs" / "colab_live_train.log"
LOCAL_LOG.parent.mkdir(parents=True, exist_ok=True)

SESSION_NAME = "mergen-snn"

print(f"Colab Monitor & Auto-Sync daemon started for session '{SESSION_NAME}'...")

def run_cmd(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_DIR))

# Helper probe script to run on remote
REMOTE_CHECK_CODE = """
import os, json, subprocess
ckpt_dir = '/content/mergen-snn/checkpoints'
files = []
if os.path.exists(ckpt_dir):
    for f in os.listdir(ckpt_dir):
        fp = os.path.join(ckpt_dir, f)
        if os.path.isfile(fp):
            files.append({'name': f, 'size': os.path.getsize(fp)})

train_running = False
ps = subprocess.run(['pgrep', '-f', 'train_lm.py'], capture_output=True, text=True)
if ps.stdout.strip():
    train_running = True

log_tail = ''
log_fp = '/content/mergen-snn/train.log'
if os.path.exists(log_fp):
    t = subprocess.run(['tail', '-n', '15', log_fp], capture_output=True, text=True)
    log_tail = t.stdout

print('---JSON_START---')
print(json.dumps({'running': train_running, 'files': files, 'log_tail': log_tail}))
print('---JSON_END---')
"""

remote_probe_file = REPO_DIR / "scripts" / "_remote_sync_probe.py"
remote_probe_file.write_text(REMOTE_CHECK_CODE, encoding="utf-8")

consecutive_failures = 0
done = False

while not done:
    try:
        # Execute probe on Colab
        res = run_cmd([
            "uv", "tool", "run", "colabapi", "raw", "--", "exec",
            "-s", SESSION_NAME, "-f", "scripts/_remote_sync_probe.py"
        ])
        
        if res.returncode != 0:
            print(f"[{time.strftime('%H:%M:%S')}] Probe failed (returncode {res.returncode}): {res.stderr[:200]}")
            consecutive_failures += 1
            if consecutive_failures > 10:
                print("Too many probe failures. Exiting monitor.")
                break
            time.sleep(20)
            continue
        
        consecutive_failures = 0
        output = res.stdout
        if "---JSON_START---" in output and "---JSON_END---" in output:
            json_part = output.split("---JSON_START---")[1].split("---JSON_END---")[0].strip()
            status = json.loads(json_part)
            
            is_running = status.get("running", False)
            remote_files = status.get("files", [])
            log_tail = status.get("log_tail", "")
            
            # Write to local live log
            with open(LOCAL_LOG, "w", encoding="utf-8") as lf:
                lf.write(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                lf.write(f"Is Running: {is_running}\n")
                lf.write(f"Remote Checkpoints: {len(remote_files)}\n\n")
                lf.write(log_tail)
            
            # Sync any new or modified checkpoint
            for rf in remote_files:
                name = rf["name"]
                size = rf["size"]
                if name.startswith("."):
                    continue
                local_path = LOCAL_CKPT_DIR / name
                
                # Check if we should download
                should_download = False
                if not local_path.exists():
                    should_download = True
                elif local_path.stat().st_size != size and not name.endswith(".png"):
                    should_download = True
                
                if should_download:
                    print(f"[{time.strftime('%H:%M:%S')}] Downloading {name} ({size / (1024*1024):.2f} MB)...")
                    dl_res = run_cmd([
                        "uv", "tool", "run", "colabapi", "raw", "--", "download",
                        "-s", SESSION_NAME,
                        f"/content/mergen-snn/checkpoints/{name}",
                        str(local_path)
                    ])
                    if dl_res.returncode == 0:
                        print(f"[{time.strftime('%H:%M:%S')}] Downloaded {name} successfully!")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] Download error for {name}: {dl_res.stderr}")
            
            # Check if finished
            if not is_running and any(rf["name"] == "small_lm_final.pt" for rf in remote_files):
                print(f"[{time.strftime('%H:%M:%S')}] Training completed and final checkpoints saved!")
                done = True
                break
                
            # Print brief status
            print(f"[{time.strftime('%H:%M:%S')}] Training running: {is_running} | Files synced: {len(remote_files)}")
            
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Exception in monitor: {e}")
    
    time.sleep(30)

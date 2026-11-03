
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
for log_candidate in ['/content/mergen-snn/runs/stage5_v3_t4_train_part2.log', '/content/mergen-snn/runs/stage5_v3_t4_train.log', '/content/mergen-snn/pipeline_runner_part2.log', '/content/mergen-snn/prep_data.log']:
    if os.path.exists(log_candidate):
        t = subprocess.run(['tail', '-n', '15', log_candidate], capture_output=True, text=True)
        log_tail = '[' + log_candidate + ']\n' + t.stdout
        break

print('---JSON_START---')
print(json.dumps({'running': train_running, 'files': files, 'log_tail': log_tail}))
print('---JSON_END---')

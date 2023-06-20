import json
import os

LOGS_DIR = 'YOUR_LOGS_DIR_HERE'

os.chdir(LOGS_DIR)
chunk_size = 20
manifest_list = []
paths = [p for p in os.listdir() if 'error' not in p and p.endswith('.json')]
chunks = []

for path in paths:
    fps = []
    with open(path, 'r') as f:
        data = json.load(f)
    for ed in data:
        for page_idx in ed['pages'].keys():
            fps.append(ed['pages'][page_idx]['page_url'])

    date = path.split('_')[1].split('.')[0]
    print(date, len(data), len(fps))
    i = 0
    
    while i + chunk_size < len(fps):
        chunks.append(fps[i:i+chunk_size])
        manifest_list.append((date, i, i + chunk_size))
        i += chunk_size

    if i != len(fps):
        chunks.append(fps[i:])
        manifest_list.append((date, i, len(fps)))

assert len(manifest_list) == len(chunks)

with open('manifest_list.txt', 'w') as f:
    for i, (date, start, end) in enumerate(manifest_list):
        f.write(f'{date},{start},{end}\n')

for i, chunk in enumerate(chunks):
    with open(f'manifest_{i}.txt', 'w') as f:
        for fp in chunk:
            f.write(fp + '\n')
    

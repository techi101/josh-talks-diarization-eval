"""Verify copied artifact hashes and the saved pilot's central numerical claims."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'PROVENANCE.json').read_text())
for relative, record in manifest.items():
    actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    assert actual == record['sha256'], f'Changed original artifact: {relative}'

rows=[]
for name in ['small_run_results.csv','pyannote_results.csv','sortformer_results.csv']:
    with (ROOT / 'results' / name).open(newline='',encoding='utf-8-sig') as f:
        rows.extend(csv.DictReader(f))
assert len(rows)==168
keys=[tuple(r[k] for k in ['meeting','mic','system','setting']) for r in rows]
assert len(keys)==len(set(keys)), 'Duplicate result rows'
groups=defaultdict(list)
for row in rows:
    assert abs(float(row['DER'])-sum(float(row[k]) for k in ['miss','fa','conf']))<0.02
    groups[(row['system'],row['setting'])].append(float(row['DER']))
assert len(groups)==28 and all(len(v)==6 for v in groups.values())
mean=lambda sys,setting:sum(groups[(sys,setting)])/6
strict='strict (0 collar, overlap scored)'
expected={
 'pyannote community-1 (auto count)':21.4,
 'pyannote community-1, told true count':24.9,
 'Sortformer v2, 30 s latency (offline-like)':31.4,
 'Sortformer v2, 1.04 s latency (live)':38.1,
 'S3 same, told true count':46.7,
 'S2 VAD + ECAPA + AHC (auto count)':49.9,
 'S1 VAD + single speaker':55.9,
}
for system,value in expected.items():
    assert round(mean(system,strict),1)==value,(system,mean(system,strict))
assert round(mean('pyannote community-1 (auto count)','0.25 s collar, overlap skipped (old NIST style)'),1)==11.4

secret_patterns=[r'hf_[A-Za-z0-9]{20,}',r'gh[pousr]_[A-Za-z0-9]{25,}',r'sk-[A-Za-z0-9]{25,}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
for p in ROOT.rglob('*'):
    if not p.is_file() or '.git' in p.parts:continue
    assert p.suffix.lower() not in {'.wav','.mp3','.flac','.ckpt','.nemo'},f'Unexpected raw data/model: {p}'
    if p.suffix.lower() in {'.py','.html','.json','.jsonc','.txt','.md'}:
        text=p.read_text(encoding='utf-8-sig')
        assert not any(re.search(pattern,text) for pattern in secret_patterns),f'Credential-like content: {p.relative_to(ROOT)}'
        if p.suffix=='.py':ast.parse(text)
print('PASS: 22 source hashes; 168 unique result rows; 7 strict means; scoring sensitivity; error components; Python syntax; credential-pattern and raw-data checks.')

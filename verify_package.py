"""Verify stored review-package bytes; this is not a scientific validity test."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
m=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
for r in m['files']:
    p=(ROOT/r['path']).resolve()
    assert ROOT in p.parents and p.is_file(),r['path']
    assert p.stat().st_size==r['bytes'],r['path']
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    assert h.hexdigest()==r['sha256'],r['path']
print(json.dumps({'status':'PASS','files':len(m['files']),'scope':'Stored bytes only; see README for scientific replay'}))

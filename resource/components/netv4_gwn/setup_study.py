# -*- coding: utf-8 -*-
"""Pin existing sources and inputs before fitting the one declared architecture."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DESKTOP = ROOT.parent
SOURCE = DESKTOP / 'NETV4_WWW_argument_revision_20261009'
UPSTREAM = DESKTOP / 'special veilcle predict/aaai_submission/AAAI2027_final_candidate/artifact/external/official_baselines/src/graphwavenet/Graph-WaveNet-6b162e80c59a1d494809252eca055cff93dc66b1'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')

assert not (ROOT / 'PINS.json').exists()
(ROOT / 'vendor').mkdir()
for name in ('model.py', 'LICENSE'):
    shutil.copy2(UPSTREAM / name, ROOT / 'vendor' / name)
archive = SOURCE / 'resource/archives/netv4_review_candidate_v0_3.zip'
with zipfile.ZipFile(archive) as z:
    manifest = json.loads(z.read('MANIFEST.json').decode('utf-8-sig'))
    expected = {r['path']: r for r in manifest['files']}
    for variant in ('packaged', 'working'):
        for name in ('history.npz', 'targets.npz', 'splits.npz'):
            key = 'data/' + variant + '/' + name
            payload = z.read(key)
            assert hashlib.sha256(payload).hexdigest() == expected[key]['sha256']
            target = ROOT / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
shutil.copytree(SOURCE / 'resource/components/netv4_mv/task', ROOT / 'task')
shutil.copy2(SOURCE / 'resource/components/netv4_mv/score.py', ROOT / 'score.py')
shutil.copy2(SOURCE / 'resource/components/netv4_mv/scores/responses.csv', ROOT / 'prior_responses.csv')
shutil.copy2(SOURCE / 'resource/components/netv4_mv/scores/accuracy.csv', ROOT / 'prior_accuracy.csv')
pins = [p for d in ('vendor', 'data', 'task') for p in (ROOT / d).rglob('*') if p.is_file()]
pins += [ROOT / name for name in ('PROTOCOL_ZH.md', 'score.py', 'prior_responses.csv', 'prior_accuracy.csv',
    'model_adapter.py', 'run_study.py', 'setup_study.py')]
dump(ROOT / 'PINS.json', {'protocol_fixed_before_fit': True, 'previous_development_exposure': True,
    'upstream_repository': 'https://github.com/nnzhan/Graph-WaveNet',
    'upstream_commit': '6b162e80c59a1d494809252eca055cff93dc66b1',
    'legacy_archive_sha256': sha(archive), 'old_candidate_unchanged': True,
    'files': [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(pins)]})
dump(ROOT / 'SCIENTIFIC_LEDGER.json', {'planned_architectures': 1, 'planned_fits': 6, 'planned_seeds': [1, 2, 3],
    'planned_training_variants': ['packaged', 'working'], 'planned_frozen_mv_predictions': 6,
    'new_generated_scenarios': 0, 'new_observed_records': 0, 'fits_completed': 0,
    'acceptance_probability_established': False})
print(json.dumps({'pins': len(pins), 'architectures': 1, 'planned_fits': 6, 'new_data': 0}))

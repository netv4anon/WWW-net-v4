# -*- coding: utf-8 -*-
"""Populate a new training directory from delivered pinned inputs and code."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''): h.update(b)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--mv-task', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    assert not out.exists(), 'Choose a new directory'
    assert out != ROOT and ROOT not in out.parents, 'Use a directory outside this component'
    pins = json.loads((ROOT / 'PINS.json').read_text(encoding='utf-8'))
    assert sha(args.archive) == pins['legacy_archive_sha256']
    out.mkdir(parents=True)
    for name in ('PINS.json', 'PROTOCOL_ZH.md', 'run_study.py', 'model_adapter.py', 'setup_study.py',
        'score.py', 'prior_responses.csv', 'prior_accuracy.csv', 'evaluate_study.py',
        'reader_adapter.py', 'predict_reader.py'):
        shutil.copy2(ROOT / name, out / name)
    shutil.copytree(ROOT / 'vendor', out / 'vendor')
    shutil.copytree(ROOT / 'technical_attempts', out / 'technical_attempts')
    shutil.copytree(args.mv_task, out / 'task')
    with zipfile.ZipFile(args.archive) as z:
        for variant in ('packaged', 'working'):
            for name in ('history.npz', 'targets.npz', 'splits.npz'):
                key = 'data/' + variant + '/' + name
                target = out / key
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(key))
    for row in pins['files']:
        assert sha(out / row['path']) == row['sha256'], row['path']
    ledger = {'planned_architectures': 1, 'planned_fits': 6, 'planned_seeds': [1, 2, 3],
        'planned_training_variants': ['packaged', 'working'], 'planned_frozen_mv_predictions': 6,
        'fits_completed': 0, 'fit_attempts_started': 2, 'inherited_technical_attempts': 2,
        'technical_superseded_completed_fits': 1, 'technical_aborted_fit_attempts': 1,
        'new_generated_scenarios': 0, 'new_observed_records': 0,
        'acceptance_probability_established': False, 'reproduction_is_not_new_independent_scientific_evidence': True}
    (out / 'SCIENTIFIC_LEDGER.json').write_text(json.dumps(ledger, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'training_staging': str(out), 'all_input_pins_verified': len(pins['files']),
        'models_fitted_by_this_command': 0, 'technical_sensitivity_inherited_not_refitted': True}))

if __name__ == '__main__': main()

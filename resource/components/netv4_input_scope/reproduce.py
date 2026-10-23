"""Portable adapter for one frozen analysis, not a new scientific replication."""
import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN = HERE / 'frozen'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--resource-root', type=Path, default=HERE.parents[1])
    ap.add_argument('--out', type=Path, required=True, help='New, nonexistent output directory.')
    args = ap.parse_args()
    resource, out = args.resource_root.resolve(), args.out.resolve()
    assert not out.exists(), 'Fresh output required; never overwrite a frozen analysis.'
    # No forecast fitting or inference is performed by either adapter.
    plan = json.loads((FROZEN / 'FROZEN_PLAN.json').read_text(encoding='utf-8'))
    for row in plan['files']:
        original = row['path'].replace('\\', '/')
        if '/resource/' in original:
            p = resource / original.split('/resource/', 1)[1]
        elif original.endswith('/PROTOCOL_ZH.md') or original.endswith('/run_audit.py'):
            p = FROZEN / original.rsplit('/', 1)[-1]
        else:
            continue  # Original history/split members are checked after archive extraction.
        assert sha(p) == row['sha256'], str(p)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(FROZEN / 'PROTOCOL_ZH.md', out / 'PROTOCOL_ZH.md')
    with tempfile.TemporaryDirectory(prefix='netv4_input_reference_') as staging_name:
        staging = Path(staging_name)
        with zipfile.ZipFile(resource / 'archives/netv4_review_candidate_v0_3.zip') as z:
            for variant in ('packaged', 'working'):
                for name in ('history.npz', 'splits.npz'):
                    member = 'data/' + variant + '/' + name
                    p = staging / member
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as src, p.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
        spec = importlib.util.spec_from_file_location('frozen_input_audit', FROZEN / 'run_audit.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT, module.RESOURCE, module.STAGING = out, resource, staging
        module.MV = resource / 'components/netv4_mv/task'
        module.GWN = resource / 'components/netv4_gwn'
        module.main()  # Unchanged policy and unchanged scorer.
    subprocess.run([sys.executable, '-B', '-X', 'utf8', str(HERE / 'describe_reference.py'),
                    '--resource-root', str(resource), '--audit-root', str(out), '--out', str(out)], check=True)
    max_csv_error = 0.
    checked_values = 0
    for name in ('COVERAGE.csv', 'PROFILE_COVERAGE.csv', 'PRIMARY_SEED_SCORES.csv',
                 'PRIMARY_LOSS_MEANS.csv', 'PRIMARY_PROFILE_LOSS_MEANS.csv', 'REFERENCE_SOURCE_COUNTS.csv'):
        with (FROZEN / name).open(encoding='utf-8', newline='') as f:
            expected = list(csv.DictReader(f))
        with (out / name).open(encoding='utf-8', newline='') as f:
            actual = list(csv.DictReader(f))
        assert len(expected) == len(actual)
        for e, a in zip(expected, actual):
            assert e.keys() == a.keys()
            for key in e:
                if e[key] == a[key]:
                    checked_values += 1
                    continue
                try:
                    difference = abs(float(e[key]) - float(a[key]))
                except ValueError:
                    raise AssertionError((name, key, e[key], a[key]))
                assert difference <= 1e-9, (name, key, difference)
                max_csv_error = max(max_csv_error, difference)
                checked_values += 1
    for variant in ('packaged', 'working'):
        with np.load(FROZEN / (variant + '_INPUT_SELECTION.npz'), allow_pickle=False) as e, \
             np.load(out / (variant + '_INPUT_SELECTION.npz'), allow_pickle=False) as a:
            assert e.files == a.files
            for key in e.files:
                if e[key].dtype.kind == 'f':
                    np.testing.assert_allclose(a[key], e[key], rtol=0, atol=1e-9)
                else:
                    np.testing.assert_array_equal(a[key], e[key])
    report = {'mechanical_reproduction_not_independent_scientific_replication': True,
              'csv_values_checked': checked_values, 'max_csv_difference': max_csv_error,
              'all_selection_arrays_match': True, 'new_fits': 0, 'new_inference': 0,
              'new_scientific_analysis_policies': 0, 'saved_forecast_reproduction_runs': 1}
    (out / 'REPRODUCTION_CHECK.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))

if __name__ == '__main__':
    main()

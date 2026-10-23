"""Replay the added core checks from the bound original archive and paired task."""
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--component', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads((HERE / 'SOURCE_BINDING.json').read_text(encoding='utf-8'))
    assert args.archive.stat().st_size == binding['archive_bytes']
    assert sha(args.archive) == binding['archive_sha256']
    args.out.mkdir(parents=True, exist_ok=False)
    base = args.out / 'original_inputs'
    wanted = []
    for variant in ('packaged', 'working'):
        wanted.extend('data/' + variant + '/' + name + '.npz'
                      for name in ('history', 'targets', 'splits', 'analysis_metadata'))
        wanted.extend('runs/' + variant + '/STID_NoCalendar_seed' + str(seed) + '/predictions.npz'
                      for seed in (1, 2, 3))
    with zipfile.ZipFile(args.archive) as archive:
        for name in wanted:
            info = archive.getinfo(name)
            if info.file_size > 50 * 1024 * 1024:
                raise ValueError('Unexpected archive member size')
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
    frozen = json.loads((HERE / 'saved_lookup/FROZEN_INPUTS.json').read_text(encoding='utf-8'))
    for variant, sources in frozen['sources'].items():
        for name in ('history.npz', 'targets.npz', 'splits.npz', 'analysis_metadata.npz'):
            assert sha(base / 'data' / variant / name) == sources[name]
        for name, value in sources['saved_references'].items():
            assert sha(base / 'runs' / variant / name / 'predictions.npz') == value
    calls = [[sys.executable, '-B', '-X', 'utf8', str(HERE / 'run_core_reuse.py'),
              '--base', str(base), '--out', str(args.out / 'lookup')],
             [sys.executable, '-B', '-X', 'utf8', str(HERE / 'score_target_envelope.py'),
              '--component', str(args.component), '--out', str(args.out / 'target_segment')]]
    for command in calls:
        subprocess.run(command, cwd=HERE, check=True)
    lookup = json.loads((args.out / 'lookup/SUMMARY.json').read_text(encoding='utf-8'))
    expected_lookup = json.loads((HERE / 'saved_lookup/SUMMARY.json').read_text(encoding='utf-8'))
    assert lookup == expected_lookup
    segment = json.loads((args.out / 'target_segment/SUMMARY.json').read_text(encoding='utf-8'))
    expected_segment = json.loads((HERE / 'saved_target_segment/SUMMARY.json').read_text(encoding='utf-8'))
    assert segment == expected_segment
    (args.out / 'VERIFICATION.json').write_text(json.dumps(
        {'core_lookup_identical': True, 'target_segment_identical': True,
         'archive_members_extracted': len(wanted), 'parent_project_or_network_or_gpu_used': False,
         'new_model_fits': 0, 'new_independent_evidence': False}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

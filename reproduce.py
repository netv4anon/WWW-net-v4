"""Run offline paper checks; write results outside this immutable resource."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CHECKS = ('inventory', 'resource', 'center', 'multisequence', 'basel', 'representation', 'evidence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='New output directory outside the resource')
    parser.add_argument('--checks', nargs='+', choices=CHECKS, default=list(CHECKS))
    parser.add_argument('--list', action='store_true', help='List checks without running them')
    args = parser.parse_args()
    if args.list:
        print('\n'.join(CHECKS))
        return 0
    if args.out is None:
        parser.error('--out NEW_DIRECTORY is required')
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error('The output directory must be outside this resource')
    if out.exists():
        parser.error('The output directory already exists; use a new directory')
    out.mkdir(parents=True)
    commands = {
        'inventory': ['verify_package.py'],
        'resource': ['resource/resource.py', 'verify'],
        'center': ['resource/resource.py', 'replay', 'netv4', '--out', str(out / 'center')],
        'multisequence': ['resource/resource.py', 'replay', 'netv4_mv', '--out', str(out / 'multisequence')],
        'basel': ['followup/baseline_confirmation/replay_resource.py'],
        'representation': ['followup/replay_followup.py', '--arrays'],
        'evidence': ['verify_evidence.py', '--out', str(out / 'evidence')],
    }
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    records = []
    for name in args.checks:
        started = time.monotonic()
        print('Running ' + name + ' ...', flush=True)
        process = subprocess.run([sys.executable, '-B', '-X', 'utf8'] + commands[name],
                                 cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
        (out / (name + '.log')).write_bytes(process.stdout)
        records.append({'check': name, 'returncode': process.returncode,
                        'seconds': round(time.monotonic() - started, 3),
                        'log': name + '.log'})
        print(name + ': ' + ('PASS' if process.returncode == 0 else 'FAIL'), flush=True)
        if process.returncode:
            print(process.stdout.decode('utf-8', errors='replace')[-4000:], flush=True)
    passed = all(record['returncode'] == 0 for record in records)
    report = {'status': 'PASS' if passed else 'FAIL', 'checks': records,
              'scope': 'Offline byte checks, retained-profile construction and saved-forecast scoring.',
              'new_training': 0, 'new_neural_inference': 0,
              'external_raw_source_audit': 'Not rerun; requires separately acquired source.',
              'historical_raw_source_regeneration': 'Not supported for the missing NYC source snapshot.'}
    (out / 'REPLAY_SUMMARY.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())

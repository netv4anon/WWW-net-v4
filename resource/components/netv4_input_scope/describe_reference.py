"""Describe already-selected nearest references; no new filter or outcome scoring."""
import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resource-root', type=Path, required=True)
    ap.add_argument('--audit-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'REFERENCE_SOURCE_COUNTS.csv').exists(), 'No overwrite.'
    archive = args.resource_root / 'archives/netv4_review_candidate_v0_3.zip'
    pins = json.loads((args.resource_root / 'components/netv4_gwn/PINS.json').read_text(encoding='utf-8'))
    pinmap = {r['path']: r['sha256'] for r in pins['files']}
    rows, bindings = [], []
    with zipfile.ZipFile(archive) as z:
        for variant in ('packaged', 'working'):
            split_name = 'data/' + variant + '/splits.npz'
            split_bytes = z.read(split_name)
            assert hashlib.sha256(split_bytes).hexdigest() == pinmap[split_name]
            with np.load(z.open(split_name), allow_pickle=False) as a:
                train = a['train'].copy()
            source_name = 'data/' + variant + '/analysis_metadata.npz'
            with np.load(z.open(source_name), allow_pickle=False) as a:
                source = a['source'].copy()
            bindings.append({'member': source_name, 'sha256': hashlib.sha256(z.read(source_name)).hexdigest()})
            tags = sorted(np.unique(source).tolist())
            for tag in tags:
                rows.append({'variant': variant, 'population': 'training_reference', 'arm': 'all',
                             'source_tag': tag, 'count': int((source[train] == tag).sum()), 'denominator': len(train)})
            with np.load(args.audit_root / (variant + '_INPUT_SELECTION.npz'), allow_pickle=False) as a:
                near = a['nearest_training_index'].copy()
                included = a['included'].copy()
                primary = a['timing'] == 'cutoff_spanning'
                arms = a['arm'].tolist()
            assert np.isin(near, train).all()
            for pop, mask in (('primary_full', primary), ('primary_included', primary & included),
                              ('primary_excluded', primary & ~included)):
                for i, arm in enumerate(arms):
                    selected_source = source[near[i, mask]]
                    for tag in tags:
                        rows.append({'variant': variant, 'population': pop, 'arm': arm, 'source_tag': tag,
                                     'count': int((selected_source == tag).sum()), 'denominator': int(mask.sum())})
    with (out / 'REFERENCE_SOURCE_COUNTS.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    seal = {'scope': 'Descriptive source composition of the frozen nearest training references only.',
            'selection_masks_changed': False, 'new_score_calculations': False,
            'observed_incident_validation': False, 'metadata_bindings': bindings,
            'nearest_neighbour_ties': 'Source tag of the fixed search returned index; no claim of unique physical neighbour.'}
    (out / 'REFERENCE_SOURCE_BINDING.json').write_text(json.dumps(seal, indent=2) + '\n', encoding='utf-8')
    print(json.dumps([r for r in rows if r['population'] == 'primary_included']))

if __name__ == '__main__':
    main()

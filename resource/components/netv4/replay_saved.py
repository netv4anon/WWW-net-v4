"""Standalone replay of all eleven saved configurations, without project imports."""
import argparse
import hashlib
import json
from pathlib import Path
from evaluate import align_prediction, dump, load_task, score, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    task = load_task(here / 'task')
    models = json.loads((here / 'MODELS.json').read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=False)
    primary = []
    for meta in models:
        path = here / meta['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('Saved prediction checksum mismatch: ' + meta['id'])
        result = score(task, align_prediction(task, path), meta['prediction_mode'])
        folder = args.out / meta['id']
        folder.mkdir()
        for name in ['cross_losses', 'decompositions', 'population_counts']:
            write_csv(folder / (name + '.csv'), result[name])
        dump(folder / 'equal_history.json', result['equal_history'])
        for row in result['decompositions']:
            if row['population'] == 'all_steps' and row['cohort'] in ['all', 'test_origin'] and row['timing_group'] == 'all':
                primary.append({'predictor': meta['id'], **row})
    write_csv(args.out / 'primary_rows.csv', primary)
    print('Replayed %d saved configurations' % len(models))


if __name__ == '__main__':
    main()

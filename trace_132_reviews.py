"""Reconcile UIT-ViSD4SA raw split counts with the input to BIO preparation.

Run from the ABSA project root:
  python trace_132_reviews.py --raw data/visd4sa --processed data/processed_v2

Uses the project's existing src.data validation and duplicate-removal functions;
does not alter datasets or train models. Saves a JSON report and rejected rows.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from src.data import read_aspects, remove_duplicates

ORIGINAL_COUNTS = {'train': 7785, 'dev': 1112, 'test': 2225}
EXPECTED_PRE_BIO = {'train': 7694, 'dev': 1098, 'test': 2198}
EXPECTED_FINAL = {'train': 7360, 'dev': 1037, 'test': 2076}


def count_nonempty_lines(path):
    with path.open(encoding='utf-8') as handle:
        return sum(bool(line.strip()) for line in handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='data/visd4sa')
    parser.add_argument('--processed', default='data/processed_v2')
    parser.add_argument('--out', default='outputs/data_audit/trace_132_reviews.json')
    args = parser.parse_args()
    raw = Path(args.raw)
    processed = Path(args.processed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    first_stage_rows = {}
    first_stage_stats = {}
    for split in ('train', 'dev', 'test'):
        path = raw / f'{split}.jsonl'
        raw_count = count_nonempty_lines(path)
        audit_path = out.parent / f'{split}_annotation_rejected.jsonl'
        rows = read_aspects(path, invalid_policy='skip', audit_path=audit_path)
        reasons = Counter()
        with audit_path.open(encoding='utf-8') as handle:
            for line in handle:
                item = json.loads(line)
                reasons[item['reason']] += 1
        first_stage_rows[split] = rows
        first_stage_stats[split] = {
            'raw': raw_count,
            'original_source_count': ORIGINAL_COUNTS[split],
            'annotation_valid': len(rows),
            'annotation_rejected': raw_count - len(rows),
            'annotation_reasons': dict(reasons),
        }

    filtered, duplicate_removed = remove_duplicates(first_stage_rows)
    duplicate_reasons = Counter(item['reason'] for item in duplicate_removed)
    duplicate_path = out.parent / 'train_duplicates_removed.jsonl'
    with duplicate_path.open('w', encoding='utf-8') as handle:
        for item in duplicate_removed:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')

    preparation_log_path = processed / 'preprocessing_report.json'
    preparation_log = json.loads(preparation_log_path.read_text(encoding='utf-8'))
    per_split = {}
    for split in ('train', 'dev', 'test'):
        stage = first_stage_stats[split]
        pre_bio = len(filtered[split])
        bio_accepted = preparation_log[f'{split}_accepted']
        bio_rejected = preparation_log[f'{split}_rejected']
        actual_final_count = count_nonempty_lines(processed / f'{split}.jsonl')
        per_split[split] = {
            **stage,
            'duplicates_removed': len(duplicate_removed) if split == 'train' else 0,
            'pre_bio_count': pre_bio,
            'expected_pre_bio_count': EXPECTED_PRE_BIO[split],
            'bio_rejected': bio_rejected,
            'bio_accepted': bio_accepted,
            'final_file_count': actual_final_count,
            'expected_final_count': EXPECTED_FINAL[split],
            'raw_minus_pre_bio': stage['raw'] - pre_bio,
            'counts_reconcile': (
                stage['raw'] == ORIGINAL_COUNTS[split]
                and pre_bio == EXPECTED_PRE_BIO[split]
                and pre_bio == bio_accepted + bio_rejected
                and bio_accepted == actual_final_count == EXPECTED_FINAL[split]
            ),
        }

    report = {
        'splits': per_split,
        'duplicate_reasons_train': dict(duplicate_reasons),
        'total_raw': sum(x['raw'] for x in per_split.values()),
        'total_before_bio': sum(x['pre_bio_count'] for x in per_split.values()),
        'total_rejected_before_bio': sum(x['raw_minus_pre_bio'] for x in per_split.values()),
        'total_rejected_at_bio': sum(x['bio_rejected'] for x in per_split.values()),
        'total_final': sum(x['final_file_count'] for x in per_split.values()),
        'all_counts_reconcile': all(x['counts_reconcile'] for x in per_split.values()),
        'notes': 'First-stage source code excludes invalid annotations and removes train duplicates; BIO-stage exclusions come from processed_v2/preprocessing_report.json. Counts reflect the current source and files, and must be reconciled with the actual historical pipeline version.',
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for split, info in per_split.items():
        print(f"{split}: raw={info['raw']} -> pre-BIO={info['pre_bio_count']} "
              f"(annotation invalid={info['annotation_rejected']}, "
              f"duplicates={info['duplicates_removed']}) -> "
              f"BIO accepted={info['bio_accepted']} (+ rejected={info['bio_rejected']}); "
              f"OK={info['counts_reconcile']}")
    print(f"TOTAL: {report['total_raw']} -> {report['total_before_bio']} -> {report['total_final']} "
          f"| before-BIO removed={report['total_rejected_before_bio']} "
          f"| BIO removed={report['total_rejected_at_bio']} "
          f"| ALL MATCH={report['all_counts_reconcile']}")
    print(f'Report saved: {out}')


if __name__ == '__main__':
    main()

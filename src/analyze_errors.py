
"""Phân tích lỗi từ kết quả dự đoán trên tập test."""

import argparse
import json
import re
from pathlib import Path

from .data import load_jsonl
from .evaluate_aspects import score as aspect_score


LABELS = ('negative', 'neutral', 'positive')

NEGATION = re.compile(
    r'(?<!\w)(?:không|chẳng|chả|chưa|chớ|ko|k|khum|hok|hông|hong|hem)(?!\w)',
    re.I
)
TEENCODE = re.compile(
    r'(?<!\w)(?:ko|khum|hok|hong|hem|đc|dc|mk|mik|mn|sp|vs|j|z|tui|nhaaa)(?!\w)',
    re.I
)
EMOJI = re.compile(r'[\U0001F000-\U0001FAFF\u2600-\u27BF]')


def heuristic_groups(text):
    # Chia nhóm câu theo đặc điểm
    return {
        'negation_heuristic': bool(NEGATION.search(text)),
        'emoji_heuristic': bool(EMOJI.search(text)),
        'teencode_heuristic': bool(TEENCODE.search(text)),
        'short_le_5_whitespace_tokens': len(text.split()) <= 5,
    }


def overall_stats(rows):
    # Tính Accuracy và Macro-F1
    n = len(rows)

    if not n:
        return {
            'n': 0,
            'accuracy': None,
            'macro_f1': None,
            'gold_counts': {k: 0 for k in LABELS},
            'wrong': 0,
        }

    correct = sum(
        row['gold'] == row['predicted']
        for row in rows
    )

    f1s = []

    for label in LABELS:
        tp = sum(
            r['gold'] == label and r['predicted'] == label
            for r in rows
        )
        fp = sum(
            r['gold'] != label and r['predicted'] == label
            for r in rows
        )
        fn = sum(
            r['gold'] == label and r['predicted'] != label
            for r in rows
        )

        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0
        f1s.append(f1)

    return {
        'n': n,
        'accuracy': correct / n,
        'macro_f1': sum(f1s) / len(f1s),
        'gold_counts': {
            k: sum(r['gold'] == k for r in rows)
            for k in LABELS
        },
        'wrong': n - correct,
    }


def aspect_stats(gold, preds):
    # Tính F1 cho span và aspect-sentiment
    if not gold:
        return {
            'n': 0,
            'exact_span': None,
            'aspect_sentiment_pair': None,
        }

    return {
        'n': len(gold),
        'exact_span': aspect_score(gold, preds, pair=False),
        'aspect_sentiment_pair': aspect_score(gold, preds, pair=True),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--overall-test',
        default='data/overall_visfd/test.jsonl'
    )
    parser.add_argument(
        '--overall-pred',
        default='outputs/evaluation/overall/predictions.jsonl'
    )
    parser.add_argument(
        '--aspect-test',
        default='data/processed_v2/test.jsonl'
    )
    parser.add_argument(
        '--aspect-pred',
        default='outputs/evaluation/aspect_kaggle/test_predictions.jsonl'
    )
    parser.add_argument(
        '--output',
        default='outputs/analysis/error_analysis.json'
    )

    args = parser.parse_args()

    # Đọc dữ liệu test và kết quả dự đoán
    overall_test = load_jsonl(args.overall_test)
    overall_pred = load_jsonl(args.overall_pred)
    aspect_test = load_jsonl(args.aspect_test)
    aspect_pred = load_jsonl(args.aspect_pred)

    # Kiểm tra số lượng dữ liệu
    if len(overall_test) != len(overall_pred):
        raise ValueError(
            'Số lượng câu Overall test và dự đoán không khớp'
        )

    if len(aspect_test) != len(aspect_pred):
        raise ValueError(
            'Số lượng câu Aspect test và dự đoán không khớp'
        )

    # Kiểm tra nhãn và thứ tự câu
    for i, (gold, pred) in enumerate(
        zip(overall_test, overall_pred)
    ):
        if (
            gold['text'] != pred['text']
            or gold['overall'] != pred['gold']
        ):
            raise ValueError(
                f'Dữ liệu hoặc nhãn Overall không khớp tại câu {i}'
            )

        if (
            pred['gold'] not in LABELS
            or pred['predicted'] not in LABELS
        ):
            raise ValueError(
                f'Có nhãn Overall không hợp lệ tại câu {i}'
            )

    for i, (gold, pred) in enumerate(
        zip(aspect_test, aspect_pred)
    ):
        if gold['text'] != pred['text']:
            raise ValueError(
                f'Dữ liệu Aspect không khớp tại câu {i}'
            )

        if (
            not isinstance(gold.get('spans'), list)
            or not isinstance(pred.get('spans'), list)
        ):
            raise ValueError(
                f'Thiếu danh sách span Aspect tại câu {i}'
            )

    # Tổng hợp kết quả
    output = {
        'notes': [
            'Nhóm phủ định, emoji và teencode được lọc bằng quy tắc.',
            'Overall và Aspect dùng hai tập test riêng.',
            'Nhóm nhiều khía cạnh và trái chiều dựa trên nhãn Aspect.',
            'Các câu mỉa mai cần được kiểm tra thủ công.',
        ],
        'overall': {
            'all': overall_stats(overall_pred),
            'groups': {},
            'first_errors': [],
        },
        'aspect': {
            'all': aspect_stats(aspect_test, aspect_pred),
            'groups': {},
            'first_errors': [],
        },
    }

    # Đánh giá theo từng nhóm câu
    for group in heuristic_groups(''):
        overall_subset = [
            row for row in overall_pred
            if heuristic_groups(row['text'])[group]
        ]

        output['overall']['groups'][group] = overall_stats(
            overall_subset
        )

        indices = [
            i for i, row in enumerate(aspect_test)
            if heuristic_groups(row['text'])[group]
        ]

        output['aspect']['groups'][group] = aspect_stats(
            [aspect_test[i] for i in indices],
            [aspect_pred[i] for i in indices],
        )

    # Câu có nhiều khía cạnh hoặc cảm xúc trái chiều
    aspect_groups = {
        'multiple_gold_aspects_ge_2': (
            lambda row: len(row['spans']) >= 2
        ),
        'mixed_gold_polarities_ge_2': (
            lambda row: len({
                span['polarity']
                for span in row['spans']
            }) >= 2
        ),
    }

    for group, condition in aspect_groups.items():
        indices = [
            i for i, row in enumerate(aspect_test)
            if condition(row)
        ]

        output['aspect']['groups'][group] = aspect_stats(
            [aspect_test[i] for i in indices],
            [aspect_pred[i] for i in indices],
        )

    # Lấy một số câu Overall dự đoán sai
    for i, row in enumerate(overall_pred):
        if (
            row['gold'] != row['predicted']
            and len(output['overall']['first_errors']) < 8
        ):
            output['overall']['first_errors'].append({
                'test_index': i,
                'text': row['text'],
                'gold': row['gold'],
                'predicted': row['predicted'],
            })

    # Lấy một số câu Aspect dự đoán sai
    for i, (gold, pred) in enumerate(
        zip(aspect_test, aspect_pred)
    ):
        gold_span = {
            (s['start'], s['end'])
            for s in gold['spans']
        }
        pred_span = {
            (s['start'], s['end'])
            for s in pred['spans']
        }

        gold_pair = {
            (
                s['start'],
                s['end'],
                s['category'],
                s['polarity'],
            )
            for s in gold['spans']
        }

        pred_pair = {
            (
                s['start'],
                s['end'],
                s['category'],
                s['polarity'],
            )
            for s in pred['spans']
        }

        if (
            gold_pair != pred_pair
            and len(output['aspect']['first_errors']) < 8
        ):
            output['aspect']['first_errors'].append({
                'test_index': i,
                'text': gold['text'],
                'gold_spans': gold['spans'],
                'predicted_spans': pred['spans'],
                'span_exact_match': gold_span == pred_span,
                'pair_exact_match': gold_pair == pred_pair,
            })

    # Lưu kết quả ra JSON
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    # In kết quả Overall
    print('\n=== PHÂN TÍCH OVERALL ===')
    print('Toàn bộ:', output['overall']['all'])

    for key, value in output['overall']['groups'].items():
        print(
            key,
            'n=', value['n'],
            'accuracy=', value['accuracy'],
            'macro_f1=', value['macro_f1'],
        )

    # In kết quả Aspect
    print('\n=== PHÂN TÍCH ASPECT ===')
    print('Toàn bộ:', output['aspect']['all'])

    for key, value in output['aspect']['groups'].items():
        span = value['exact_span'] or {}
        pair = value['aspect_sentiment_pair'] or {}

        print(
            key,
            'n=', value['n'],
            'span_f1=', span.get('f1'),
            'pair_f1=', pair.get('f1'),
        )

    print('\nĐã lưu kết quả tại:', path)
    print('Lưu ý: các nhóm heuristic cần kiểm tra lại khi viết paper.')


if __name__ == '__main__':
    main()

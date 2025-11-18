#!/usr/bin/env python3
"""
OpenAI 배치 라벨링 결과를 CSV로 변환하는 스크립트
"""

import json
import csv
from pathlib import Path


def export_openai_results_to_csv(
    original_file: str,
    output_file: str,
    result_file: str,
    csv_output: str
):
    """OpenAI 결과를 원본 데이터와 결합하여 CSV로 저장합니다."""

    print(f"\n{'='*60}")
    print("OpenAI 배치 결과 CSV 변환")
    print(f"{'='*60}\n")

    # 원본 데이터 로드
    print(f"원본 데이터 로딩: {original_file}")
    with open(original_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    print(f"  -> {len(articles):,}개 기사 로드됨")

    # OpenAI 결과 로드
    print(f"\nOpenAI 결과 로딩: {result_file}")
    labels = {}

    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            custom_id = record.get('custom_id')

            # 라벨 추출
            label = None
            try:
                response = record.get('response', {})
                if response.get('status_code') == 200:
                    body = response.get('body', {})
                    choices = body.get('choices', [])
                    if choices:
                        content = choices[0].get('message', {}).get('content', '').strip()
                        if content in ['0', '1', '2']:
                            label = int(content)
            except:
                pass

            labels[custom_id] = label

    print(f"  -> {len(labels):,}개 라벨 로드됨")

    # CSV 생성
    print(f"\nCSV 파일 생성 중: {csv_output}")

    with open(csv_output, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = ['article_id', 'title', 'summary', 'topic', 'label_openai', 'label_name']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        label_names = {
            0: '옹호',
            1: '중립',
            2: '비판',
            None: '누락'
        }

        for idx, article in enumerate(articles):
            custom_id = f"article-{idx}"
            label = labels.get(custom_id)

            writer.writerow({
                'article_id': idx,
                'title': article.get('title', ''),
                'summary': article.get('summary', ''),
                'topic': article.get('topic', ''),
                'label_openai': label if label is not None else '',
                'label_name': label_names.get(label, '누락')
            })

    print(f"  -> CSV 저장 완료!")

    # 통계
    label_counts = {'0': 0, '1': 0, '2': 0, 'missing': 0}
    for label in labels.values():
        if label is not None:
            label_counts[str(label)] += 1
        else:
            label_counts['missing'] += 1

    print(f"\n{'='*60}")
    print("[라벨 통계]")
    print(f"{'='*60}")
    print(f"옹호 (0): {label_counts['0']:,}개")
    print(f"중립 (1): {label_counts['1']:,}개")
    print(f"비판 (2): {label_counts['2']:,}개")
    print(f"누락: {label_counts['missing']:,}개")

    print(f"\n{'='*60}")
    print("변환 완료!")
    print(f"{'='*60}")
    print(f"\n엑셀에서 열기: {csv_output}")
    print()


def main():
    """메인 함수"""

    base_dir = Path(__file__).parent.parent.parent.parent

    # 파일 경로
    original_file = base_dir / "data" / "raw" / "bigkinds_summarized.json"
    result_file = base_dir / "data" / "batch_results" / "openai" / "batch_690f0cdf4e848190a8b4a242f2847c47_output.jsonl"
    csv_output = base_dir / "data" / "batch_results" / "openai" / "openai_labeled_results.csv"

    if not original_file.exists():
        print(f"오류: 원본 파일을 찾을 수 없습니다: {original_file}")
        return

    if not result_file.exists():
        print(f"오류: 결과 파일을 찾을 수 없습니다: {result_file}")
        return

    export_openai_results_to_csv(
        str(original_file),
        str(result_file),
        str(result_file),
        str(csv_output)
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
검수 연습용 샘플 데이터 추출 스크립트
"""

import json
import random
from pathlib import Path
import csv


def extract_samples(disagreements_file: str, output_dir: str, sample_size: int = 50):
    """검수 연습용 샘플을 추출합니다."""

    print(f"\n{'='*60}")
    print("검수 연습용 샘플 추출")
    print(f"{'='*60}\n")

    # 불일치 데이터 로드
    with open(disagreements_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"전체 불일치 데이터: {len(data):,}건")

    # 패턴별로 분류
    patterns = {
        '옹호→중립': [],
        '비판→중립': [],
        '중립→비판': [],
        '옹호→비판': [],
        '비판→옹호': [],
        '중립→옹호': [],
    }

    pattern_map = {
        (0, 1): '옹호→중립',
        (2, 1): '비판→중립',
        (1, 2): '중립→비판',
        (0, 2): '옹호→비판',
        (2, 0): '비판→옹호',
        (1, 0): '중립→옹호',
    }

    for item in data:
        openai = item.get('label_openai')
        claude = item.get('label_claude')
        key = (openai, claude)
        pattern = pattern_map.get(key)
        if pattern:
            patterns[pattern].append(item)

    # 각 패턴별 샘플 추출
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 1. 균형잡힌 샘플 (각 패턴에서 고르게)
    balanced_samples = []
    per_pattern = sample_size // len(patterns)

    for pattern, items in patterns.items():
        if items:
            n = min(per_pattern, len(items))
            balanced_samples.extend(random.sample(items, n))

    # 총 50개 맞추기
    if len(balanced_samples) < sample_size:
        remaining = sample_size - len(balanced_samples)
        all_items = [item for items in patterns.values() for item in items]
        additional = random.sample(
            [x for x in all_items if x not in balanced_samples],
            min(remaining, len(all_items) - len(balanced_samples))
        )
        balanced_samples.extend(additional)

    random.shuffle(balanced_samples)

    # JSON 저장
    balanced_json = output_path / "practice_sample_50.json"
    with open(balanced_json, 'w', encoding='utf-8') as f:
        json.dump(balanced_samples[:sample_size], f, ensure_ascii=False, indent=2)

    # CSV 저장
    balanced_csv = output_path / "practice_sample_50.csv"
    with open(balanced_csv, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = [
            'article_id', 'title', 'summary', 'topic',
            'label_openai', 'label_claude', 'final_label',
            'reviewed', 'reviewer', 'review_notes'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(balanced_samples[:sample_size])

    print(f"\n[OK] 균형잡힌 샘플 {sample_size}건 생성:")
    print(f"   JSON: {balanced_json}")
    print(f"   CSV:  {balanced_csv}")

    # 패턴별 분포 출력
    print(f"\n패턴별 분포:")
    for pattern, items in patterns.items():
        count = sum(1 for x in balanced_samples[:sample_size]
                   if (x['label_openai'], x['label_claude']) in
                   [(k, v) for (k, v), p in pattern_map.items() if p == pattern])
        if count > 0:
            print(f"  {pattern}: {count}건")

    # 2. 우선순위 1 (옹호↔비판) 전체
    priority1 = patterns['옹호→비판'] + patterns['비판→옹호']

    if priority1:
        priority1_json = output_path / "priority1_all.json"
        with open(priority1_json, 'w', encoding='utf-8') as f:
            json.dump(priority1, f, ensure_ascii=False, indent=2)

        priority1_csv = output_path / "priority1_all.csv"
        with open(priority1_csv, 'w', encoding='utf-8-sig', newline='') as f:
            fieldnames = [
                'article_id', 'title', 'summary', 'topic',
                'label_openai', 'label_claude', 'final_label',
                'reviewed', 'reviewer', 'review_notes'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(priority1)

        print(f"\n[OK] 우선순위 1 (옹호<->비판) {len(priority1)}건:")
        print(f"   JSON: {priority1_json}")
        print(f"   CSV:  {priority1_csv}")

    print(f"\n{'='*60}")
    print("샘플 추출 완료!")
    print(f"{'='*60}\n")

    print("다음 단계:")
    print("1. practice_sample_50.csv로 팀원들과 연습")
    print("2. 각자 검수 후 결과 비교")
    print("3. 일치율 80% 이상 확인")
    print("4. 본격 검수 시작\n")


def main():
    base_dir = Path(__file__).parent.parent.parent

    disagreements_file = base_dir / "data" / "review" / "disagreements_for_review.json"
    output_dir = base_dir / "data" / "review" / "samples"

    if not disagreements_file.exists():
        print(f"오류: {disagreements_file}를 찾을 수 없습니다.")
        print("먼저 extract_disagreements.py를 실행하세요.")
        return

    extract_samples(str(disagreements_file), str(output_dir), sample_size=50)


if __name__ == "__main__":
    main()

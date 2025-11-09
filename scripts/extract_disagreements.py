#!/usr/bin/env python3
"""
OpenAI와 Claude 라벨 불일치 데이터 추출 스크립트
팀원 검수를 위한 불일치 데이터를 추출합니다.
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List


def analyze_disagreements(dataset_file: str) -> Dict:
    """불일치 데이터를 분석하고 추출합니다."""

    print(f"\n{'='*60}")
    print("OpenAI vs Claude 라벨 불일치 분석")
    print(f"{'='*60}\n")

    print(f"데이터셋 로딩: {dataset_file}")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  -> {len(data):,}개 기사 로드됨\n")

    # 통계
    stats = {
        'total': len(data),
        'both_labeled': 0,
        'agreed': 0,
        'disagreed': 0,
        'only_openai': 0,
        'only_claude': 0,
        'neither': 0,
        'disagreement_patterns': Counter(),
        'disagreements': []
    }

    for article in data:
        openai_label = article.get('label_openai')
        claude_label = article.get('label_claude')

        # 라벨 상태 확인
        if openai_label is not None and claude_label is not None:
            stats['both_labeled'] += 1

            if openai_label == claude_label:
                stats['agreed'] += 1
            else:
                stats['disagreed'] += 1

                # 불일치 패턴 기록
                pattern = f"{openai_label} -> {claude_label}"
                stats['disagreement_patterns'][pattern] += 1

                # 불일치 데이터 저장
                stats['disagreements'].append({
                    'article_id': article.get('article_id'),
                    'title': article.get('title'),
                    'summary': article.get('summary'),
                    'topic': article.get('topic'),
                    'label_openai': openai_label,
                    'label_claude': claude_label,
                    'final_label': None,
                    'reviewed': False,
                    'reviewer': '',
                    'review_notes': ''
                })

        elif openai_label is not None:
            stats['only_openai'] += 1
        elif claude_label is not None:
            stats['only_claude'] += 1
        else:
            stats['neither'] += 1

    return stats


def print_statistics(stats: Dict):
    """통계를 출력합니다."""

    print(f"{'='*60}")
    print("[전체 통계]")
    print(f"{'='*60}")
    print(f"총 기사 수: {stats['total']:,}개\n")

    print(f"양쪽 모두 라벨링됨: {stats['both_labeled']:,}개")
    print(f"  - 일치: {stats['agreed']:,}개 ({stats['agreed']/stats['both_labeled']*100:.2f}%)")
    print(f"  - 불일치: {stats['disagreed']:,}개 ({stats['disagreed']/stats['both_labeled']*100:.2f}%)")
    print()

    if stats['only_openai'] > 0:
        print(f"OpenAI만 라벨링: {stats['only_openai']:,}개")
    if stats['only_claude'] > 0:
        print(f"Claude만 라벨링: {stats['only_claude']:,}개")
    if stats['neither'] > 0:
        print(f"둘 다 라벨링 안됨: {stats['neither']:,}개")

    # 불일치 패턴 분석
    if stats['disagreement_patterns']:
        print(f"\n{'='*60}")
        print("[불일치 패턴 분석]")
        print(f"{'='*60}")

        label_names = {
            0: '옹호',
            1: '중립',
            2: '비판'
        }

        print(f"\n{'OpenAI -> Claude':<20} {'건수':<10} {'비율':<10}")
        print("-" * 40)

        for pattern, count in stats['disagreement_patterns'].most_common():
            openai_label, claude_label = pattern.split(' -> ')
            openai_name = label_names[int(openai_label)]
            claude_name = label_names[int(claude_label)]
            percentage = count / stats['disagreed'] * 100

            print(f"{openai_name} -> {claude_name:<10} {count:<10} {percentage:.1f}%")


def save_disagreements(disagreements: List[Dict], output_file: str):
    """불일치 데이터를 검수용 파일로 저장합니다."""

    print(f"\n{'='*60}")
    print("[검수용 파일 저장]")
    print(f"{'='*60}")

    # 검수용 JSON 파일
    json_file = output_file
    print(f"\nJSON 파일 저장: {json_file}")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(disagreements, f, ensure_ascii=False, indent=2)
    print(f"  -> {len(disagreements):,}개 불일치 데이터 저장됨")

    # 검수용 CSV 파일 (엑셀에서 보기 편함)
    csv_file = Path(output_file).with_suffix('.csv')
    print(f"\nCSV 파일 저장: {csv_file}")

    import csv
    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = [
            'article_id', 'title', 'summary', 'topic',
            'label_openai', 'label_claude', 'final_label',
            'reviewed', 'reviewer', 'review_notes'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(disagreements)

    print(f"  -> CSV 파일 저장 완료 (엑셀로 열기 가능)")

    print(f"\n{'='*60}")
    print("검수 가이드")
    print(f"{'='*60}")
    print(f"""
1. 검수 방법:
   - CSV 파일을 엑셀로 열어서 검수하거나
   - JSON 파일을 사용하여 커스텀 검수 도구 활용

2. 검수 프로세스:
   - title, summary를 읽고 기사 논조 판단
   - OpenAI 라벨(label_openai)과 Claude 라벨(label_claude) 확인
   - 올바른 라벨을 final_label에 입력 (0, 1, 2)
   - reviewed를 true로 변경
   - reviewer에 검수자 이름 입력
   - 필요시 review_notes에 메모 작성

3. 라벨 의미:
   - 0: 옹호 (해당 정책이나 인물을 긍정적으로 서술)
   - 1: 중립 (객관적 사실 전달, 균형잡힌 시각)
   - 2: 비판 (해당 정책이나 인물을 부정적으로 서술)

4. 검수 완료 후:
   - 검수된 데이터를 최종 데이터셋에 병합
   - 모델 학습용 데이터셋으로 변환
""")


def main():
    """메인 함수"""

    base_dir = Path(__file__).parent.parent

    # 파일 경로
    dataset_file = base_dir / "data" / "labeled_dataset.json"
    disagreements_file = base_dir / "data" / "disagreements_for_review.json"

    if not dataset_file.exists():
        print(f"\n오류: 데이터셋 파일을 찾을 수 없습니다: {dataset_file}")
        print("먼저 merge_labels.py를 실행하여 데이터셋을 생성하세요.")
        return

    # 분석 실행
    stats = analyze_disagreements(str(dataset_file))

    # 통계 출력
    print_statistics(stats)

    # Claude 라벨이 없는 경우
    if stats['both_labeled'] == 0:
        print(f"\n{'='*60}")
        print("[주의]")
        print(f"{'='*60}")
        print("\nClaude API 라벨이 아직 없습니다.")
        print("Claude API로 라벨링 후 merge_labels.py를 재실행하세요.")
        print()
        return

    # 불일치 데이터 저장
    if stats['disagreed'] > 0:
        save_disagreements(stats['disagreements'], str(disagreements_file))
    else:
        print(f"\n{'='*60}")
        print("축하합니다! 모든 라벨이 일치합니다!")
        print(f"{'='*60}\n")

    print(f"\n{'='*60}")
    print("분석 완료!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
OpenAI Batch API 라벨링 결과 분석 스크립트
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

def analyze_batch_results(output_file: str) -> Dict:
    """배치 처리 결과 파일을 분석합니다."""

    results = {
        'total_records': 0,
        'successful': 0,
        'failed': 0,
        'label_distribution': Counter(),
        'invalid_labels': [],
        'errors': [],
        'status_codes': Counter(),
    }

    print(f"\n{'='*60}")
    print(f"배치 라벨링 결과 분석")
    print(f"{'='*60}\n")
    print(f"분석 파일: {output_file}")

    with open(output_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            results['total_records'] += 1

            try:
                record = json.loads(line.strip())
                custom_id = record.get('custom_id', 'unknown')
                response = record.get('response', {})
                status_code = response.get('status_code')
                error = record.get('error') or response.get('error')

                results['status_codes'][status_code] += 1

                # 에러 체크
                if error:
                    results['failed'] += 1
                    results['errors'].append({
                        'custom_id': custom_id,
                        'error': error,
                        'line': line_num
                    })
                    continue

                # 정상 처리된 경우
                if status_code == 200:
                    body = response.get('body', {})
                    choices = body.get('choices', [])

                    if choices:
                        content = choices[0].get('message', {}).get('content', '').strip()

                        # 라벨 추출 및 검증
                        if content in ['0', '1', '2']:
                            results['successful'] += 1
                            results['label_distribution'][content] += 1
                        else:
                            results['invalid_labels'].append({
                                'custom_id': custom_id,
                                'label': content,
                                'line': line_num
                            })
                    else:
                        results['failed'] += 1
                        results['errors'].append({
                            'custom_id': custom_id,
                            'error': 'No choices in response',
                            'line': line_num
                        })
                else:
                    results['failed'] += 1

            except json.JSONDecodeError as e:
                results['failed'] += 1
                results['errors'].append({
                    'custom_id': 'parse_error',
                    'error': f'JSON decode error: {str(e)}',
                    'line': line_num
                })

    return results

def print_analysis(results: Dict):
    """분석 결과를 출력합니다."""

    print(f"\n{'='*60}")
    print(f"[전체 통계]")
    print(f"{'='*60}")
    print(f"총 레코드 수: {results['total_records']:,}")
    print(f"성공: {results['successful']:,} ({results['successful']/results['total_records']*100:.2f}%)")
    print(f"실패: {results['failed']:,} ({results['failed']/results['total_records']*100:.2f}%)")

    print(f"\n{'='*60}")
    print(f"[라벨 분포]")
    print(f"{'='*60}")

    label_names = {
        '0': '옹호 (Supportive)',
        '1': '중립 (Neutral)',
        '2': '비판 (Critical)'
    }

    total_labeled = sum(results['label_distribution'].values())

    for label in ['0', '1', '2']:
        count = results['label_distribution'][label]
        percentage = (count / total_labeled * 100) if total_labeled > 0 else 0
        bar_length = int(percentage / 2)  # 50% = 25 chars
        bar = '#' * bar_length

        print(f"{label_names[label]:25} : {count:5,} ({percentage:5.2f}%) {bar}")

    print(f"\n총 라벨링된 기사: {total_labeled:,}")

    # HTTP 상태 코드 분포
    print(f"\n{'='*60}")
    print(f"[HTTP 상태 코드 분포]")
    print(f"{'='*60}")
    for status_code, count in results['status_codes'].most_common():
        print(f"Status {status_code}: {count:,}")

    # 유효하지 않은 라벨
    if results['invalid_labels']:
        print(f"\n{'='*60}")
        print(f"[유효하지 않은 라벨] ({len(results['invalid_labels'])}개)")
        print(f"{'='*60}")
        for item in results['invalid_labels'][:10]:  # 처음 10개만 표시
            print(f"Line {item['line']}: {item['custom_id']} -> '{item['label']}'")
        if len(results['invalid_labels']) > 10:
            print(f"... 그 외 {len(results['invalid_labels']) - 10}개")

    # 에러
    if results['errors']:
        print(f"\n{'='*60}")
        print(f"[에러] ({len(results['errors'])}개)")
        print(f"{'='*60}")
        for item in results['errors'][:10]:  # 처음 10개만 표시
            print(f"Line {item['line']}: {item['custom_id']}")
            print(f"  Error: {item['error']}")
        if len(results['errors']) > 10:
            print(f"... 그 외 {len(results['errors']) - 10}개")

    # 품질 평가
    print(f"\n{'='*60}")
    print(f"[품질 평가]")
    print(f"{'='*60}")

    success_rate = (results['successful'] / results['total_records'] * 100) if results['total_records'] > 0 else 0

    if success_rate >= 99:
        quality = "매우 우수 *****"
    elif success_rate >= 95:
        quality = "우수 ****"
    elif success_rate >= 90:
        quality = "양호 ***"
    elif success_rate >= 80:
        quality = "보통 **"
    else:
        quality = "개선 필요 *"

    print(f"성공률: {success_rate:.2f}%")
    print(f"품질 등급: {quality}")

    # 라벨 균형도 확인
    if total_labeled > 0:
        label_percentages = [
            results['label_distribution']['0'] / total_labeled * 100,
            results['label_distribution']['1'] / total_labeled * 100,
            results['label_distribution']['2'] / total_labeled * 100
        ]

        max_diff = max(label_percentages) - min(label_percentages)

        print(f"\n라벨 균형도:")
        if max_diff < 20:
            print(f"  매우 균형적 (최대 차이: {max_diff:.1f}%)")
        elif max_diff < 40:
            print(f"  적당히 균형적 (최대 차이: {max_diff:.1f}%)")
        else:
            print(f"  불균형 (최대 차이: {max_diff:.1f}%)")
            print(f"  → 특정 라벨이 과도하게 많거나 적을 수 있습니다.")

def main():
    """메인 함수"""

    # 파일 경로
    output_file = Path(__file__).parent.parent / "data" / "batch_690f0cdf4e848190a8b4a242f2847c47_output.jsonl"

    if not output_file.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {output_file}")
        return

    # 분석 실행
    results = analyze_batch_results(str(output_file))

    # 결과 출력
    print_analysis(results)

    print(f"\n{'='*60}")
    print("분석 완료!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
원본 데이터와 OpenAI/Claude 라벨링 결과를 결합하는 스크립트
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_original_data(file_path: str) -> List[Dict]:
    """원본 데이터를 로드합니다."""
    print(f"원본 데이터 로딩: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  -> {len(data):,}개 기사 로드됨")
    return data


def load_batch_results(file_path: str) -> Dict[str, Optional[int]]:
    """OpenAI 배치 결과 파일을 로드하여 {custom_id: label} 딕셔너리로 반환합니다."""
    print(f"OpenAI 배치 결과 로딩: {file_path}")

    labels = {}
    success_count = 0
    fail_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
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
                            success_count += 1
                        else:
                            fail_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"  경고: {custom_id} 처리 중 에러: {e}")
                fail_count += 1

            labels[custom_id] = label

    print(f"  -> 성공: {success_count:,}개, 실패: {fail_count:,}개")
    return labels


def load_claude_batch_results(file_path: str) -> Dict[str, Optional[int]]:
    """Claude 배치 결과 파일을 로드하여 {custom_id: label} 딕셔너리로 반환합니다."""
    print(f"Claude 배치 결과 로딩: {file_path}")

    labels = {}
    success_count = 0
    fail_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            custom_id = record.get('custom_id')

            # 라벨 추출 (Claude API 응답 형식)
            label = None
            try:
                result = record.get('result', {})
                if result.get('type') == 'succeeded':
                    message = result.get('message', {})
                    content_blocks = message.get('content', [])
                    if content_blocks and len(content_blocks) > 0:
                        content = content_blocks[0].get('text', '').strip()
                        if content in ['0', '1', '2']:
                            label = int(content)
                            success_count += 1
                        else:
                            fail_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"  경고: {custom_id} 처리 중 에러: {e}")
                fail_count += 1

            labels[custom_id] = label

    print(f"  -> 성공: {success_count:,}개, 실패: {fail_count:,}개")
    return labels


def merge_data(
    original_data: List[Dict],
    openai_labels: Dict[str, Optional[int]],
    claude_labels: Optional[Dict[str, Optional[int]]] = None
) -> List[Dict]:
    """원본 데이터와 라벨을 결합합니다."""
    print("\n데이터 결합 중...")

    merged = []

    for idx, article in enumerate(original_data):
        custom_id = f"article-{idx}"

        # 기본 정보 복사
        merged_article = {
            'article_id': idx,
            **article  # 원본 데이터의 모든 필드 포함
        }

        # OpenAI 라벨 추가
        openai_label = openai_labels.get(custom_id)
        merged_article['label_openai'] = openai_label

        # Claude 라벨 추가 (있는 경우)
        if claude_labels:
            claude_label = claude_labels.get(custom_id)
            merged_article['label_claude'] = claude_label

            # 일치 여부 확인
            if openai_label is not None and claude_label is not None:
                merged_article['agreement'] = (openai_label == claude_label)
            else:
                merged_article['agreement'] = None

        # 최종 라벨 (나중에 수동 검수 후 업데이트)
        merged_article['final_label'] = None
        merged_article['reviewed'] = False
        merged_article['review_notes'] = ""

        merged.append(merged_article)

    print(f"  -> {len(merged):,}개 기사 결합 완료")
    return merged


def save_merged_data(data: List[Dict], output_path: str):
    """결합된 데이터를 저장합니다."""
    print(f"\n결합 데이터 저장: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  -> 저장 완료!")


def print_statistics(data: List[Dict]):
    """데이터 통계를 출력합니다."""
    print(f"\n{'='*60}")
    print("[결합 데이터 통계]")
    print(f"{'='*60}")

    total = len(data)

    # OpenAI 라벨 통계
    openai_labeled = sum(1 for d in data if d['label_openai'] is not None)
    openai_missing = total - openai_labeled

    print(f"\n[OpenAI 라벨]")
    print(f"  라벨링됨: {openai_labeled:,}개 ({openai_labeled/total*100:.2f}%)")
    print(f"  누락: {openai_missing:,}개")

    if openai_labeled > 0:
        label_counts = {'0': 0, '1': 0, '2': 0}
        for d in data:
            if d['label_openai'] is not None:
                label_counts[str(d['label_openai'])] += 1

        print(f"\n  라벨 분포:")
        print(f"    옹호 (0): {label_counts['0']:,}개 ({label_counts['0']/openai_labeled*100:.2f}%)")
        print(f"    중립 (1): {label_counts['1']:,}개 ({label_counts['1']/openai_labeled*100:.2f}%)")
        print(f"    비판 (2): {label_counts['2']:,}개 ({label_counts['2']/openai_labeled*100:.2f}%)")

    # Claude 라벨 통계 (있는 경우)
    if any('label_claude' in d for d in data):
        claude_labeled = sum(1 for d in data if d.get('label_claude') is not None)
        claude_missing = total - claude_labeled

        print(f"\n[Claude 라벨]")
        print(f"  라벨링됨: {claude_labeled:,}개 ({claude_labeled/total*100:.2f}%)")
        print(f"  누락: {claude_missing:,}개")

        if claude_labeled > 0:
            label_counts = {'0': 0, '1': 0, '2': 0}
            for d in data:
                if d.get('label_claude') is not None:
                    label_counts[str(d['label_claude'])] += 1

            print(f"\n  라벨 분포:")
            print(f"    옹호 (0): {label_counts['0']:,}개 ({label_counts['0']/claude_labeled*100:.2f}%)")
            print(f"    중립 (1): {label_counts['1']:,}개 ({label_counts['1']/claude_labeled*100:.2f}%)")
            print(f"    비판 (2): {label_counts['2']:,}개 ({label_counts['2']/claude_labeled*100:.2f}%)")

        # 일치율 통계
        agreements = [d for d in data if d.get('agreement') is not None]
        if agreements:
            agreed = sum(1 for d in agreements if d['agreement'])
            disagreed = len(agreements) - agreed

            print(f"\n[라벨 일치도]")
            print(f"  일치: {agreed:,}개 ({agreed/len(agreements)*100:.2f}%)")
            print(f"  불일치: {disagreed:,}개 ({disagreed/len(agreements)*100:.2f}%)")

    print(f"\n{'='*60}")


def main():
    """메인 함수"""
    base_dir = Path(__file__).parent.parent

    # 파일 경로 설정
    original_file = base_dir / "data" / "bigkinds_summarized.json"
    openai_output_file = base_dir / "data" / "batch_690f0cdf4e848190a8b4a242f2847c47_output.jsonl"
    merged_output_file = base_dir / "data" / "labeled_dataset.json"

    print(f"\n{'='*60}")
    print("원본 데이터와 라벨 결합 스크립트")
    print(f"{'='*60}\n")

    # 1. 원본 데이터 로드
    original_data = load_original_data(str(original_file))

    # 2. OpenAI 결과 로드
    openai_labels = load_batch_results(str(openai_output_file))

    # 3. Claude 결과 로드 (있는 경우)
    claude_labels = None
    claude_output_file = base_dir / "data" / "batch_claude_output.jsonl"
    if claude_output_file.exists():
        print("\nClaude API 결과 파일 발견!")
        claude_labels = load_claude_batch_results(str(claude_output_file))
    else:
        print(f"\nClaude API 결과 파일 없음 (나중에 추가 가능)")
        print(f"예상 경로: {claude_output_file}")

    # 4. 데이터 결합
    merged_data = merge_data(original_data, openai_labels, claude_labels)

    # 5. 통계 출력
    print_statistics(merged_data)

    # 6. 결과 저장
    save_merged_data(merged_data, str(merged_output_file))

    print(f"\n{'='*60}")
    print("작업 완료!")
    print(f"{'='*60}")
    print(f"\n출력 파일: {merged_output_file}")
    print(f"\n다음 단계:")
    print(f"  1. Claude API로 같은 데이터 라벨링")
    print(f"  2. 이 스크립트 재실행하여 Claude 라벨 추가")
    print(f"  3. 불일치 데이터 추출 및 검수")
    print()


if __name__ == "__main__":
    main()

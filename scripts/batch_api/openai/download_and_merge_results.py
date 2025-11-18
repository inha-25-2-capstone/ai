#!/usr/bin/env python3
"""
완료된 배치 결과 다운로드 및 CSV로 병합
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

try:
    import openai
except ImportError:
    print("ERROR: OpenAI 패키지가 설치되지 않았습니다.")
    exit(1)

load_dotenv()

BASE_DIR = Path(__file__).parent.parent.parent.parent


def download_all_completed_batches():
    """완료된 모든 배치 결과 다운로드"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return None

    client = openai.OpenAI(api_key=api_key)

    print(f"\n{'='*80}")
    print("완료된 배치 결과 다운로드")
    print(f"{'='*80}\n")

    # 모든 배치 가져오기
    batches = client.batches.list(limit=100)

    # 완료된 배치만 필터링
    completed_batches = [b for b in batches.data if b.status == "completed"]

    print(f"완료된 배치: {len(completed_batches)}개\n")

    # 결과 저장 디렉토리
    results_dir = BASE_DIR / "data" / "batch_results" / "openai" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for idx, batch in enumerate(completed_batches, 1):
        print(f"{idx}/{len(completed_batches)} - 배치 ID: {batch.id}")

        if not batch.output_file_id:
            print(f"  건너뜀: 출력 파일 없음")
            continue

        # 결과 파일 다운로드
        try:
            file_response = client.files.content(batch.output_file_id)

            # JSONL 파싱
            results_text = file_response.text

            for line in results_text.strip().split('\n'):
                if line:
                    result = json.loads(line)
                    all_results.append(result)

            print(f"  OK: {len(results_text.strip().split(chr(10)))}개 결과")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\n총 결과 수: {len(all_results)}개")

    return all_results


def parse_results_to_dataframe(results):
    """결과를 DataFrame으로 변환"""
    print(f"\n{'='*80}")
    print("결과 파싱 중...")
    print(f"{'='*80}\n")

    parsed_data = []

    for result in results:
        try:
            custom_id = result.get('custom_id')
            response = result.get('response', {})

            if response.get('status_code') != 200:
                print(f"  건너뜀: {custom_id} - 응답 오류")
                continue

            body = response.get('body', {})
            choices = body.get('choices', [])

            if not choices:
                print(f"  건너뜀: {custom_id} - 응답 없음")
                continue

            # 라벨 추출
            label_text = choices[0].get('message', {}).get('content', '').strip()

            # 숫자만 추출 (0, 1, 2)
            try:
                label = int(label_text)
                if label not in [0, 1, 2]:
                    print(f"  건너뜀: {custom_id} - 잘못된 라벨: {label_text}")
                    continue
            except ValueError:
                print(f"  건너뜀: {custom_id} - 라벨 파싱 실패: {label_text}")
                continue

            parsed_data.append({
                'doc_id': custom_id,
                'label': label
            })

        except Exception as e:
            print(f"  오류: {e}")
            continue

    print(f"\n파싱 완료: {len(parsed_data)}개")

    return pd.DataFrame(parsed_data)


def merge_with_original_data(labels_df):
    """원본 데이터와 병합하여 파인튜닝용 데이터셋 생성"""
    print(f"\n{'='*80}")
    print("원본 데이터와 병합 중...")
    print(f"{'='*80}\n")

    # 원본 데이터 로드
    data_path = BASE_DIR / "data" / "add_dataset.json"

    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 정치 카테고리 데이터만 필터링
    politics_data = [
        item for item in dataset['data']
        if item.get('doc_class', {}).get('code') == '정치'
    ]

    print(f"원본 정치 데이터: {len(politics_data)}개")

    # DataFrame으로 변환
    original_data = []

    for item in politics_data:
        doc_id = item.get('doc_id')
        doc_title = item.get('doc_title', '')

        # context 추출
        contexts = []
        for para in item.get('paragraphs', []):
            context = para.get('context', '')
            if context:
                contexts.append(context)

        content = '\n\n'.join(contexts)

        original_data.append({
            'doc_id': doc_id,
            'title': doc_title,
            'content': content
        })

    original_df = pd.DataFrame(original_data)

    print(f"원본 DataFrame: {len(original_df)}행")

    # 라벨과 병합
    merged_df = original_df.merge(labels_df, on='doc_id', how='inner')

    print(f"병합 완료: {len(merged_df)}행")
    print(f"라벨 분포:")
    print(merged_df['label'].value_counts().sort_index())

    return merged_df


def save_to_csv(df, output_path=None):
    """CSV 파일로 저장"""
    if output_path is None:
        output_path = BASE_DIR / "data" / "batch_results" / "politics_labeled_dataset.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 파인튜닝에 필요한 컬럼만 선택
    final_df = df[['doc_id', 'title', 'content', 'label']]

    # CSV 저장
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*80}")
    print(f"CSV 파일 저장 완료!")
    print(f"{'='*80}")
    print(f"파일 경로: {output_path}")
    print(f"총 행 수: {len(final_df)}개")
    print(f"컬럼: {list(final_df.columns)}")
    print(f"파일 크기: {output_path.stat().st_size / (1024*1024):.2f} MB")

    # 라벨 분포 출력
    print(f"\n라벨 분포:")
    label_counts = final_df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = ['옹호', '중립', '비판'][label]
        print(f"  {label} ({label_name}): {count}개 ({count/len(final_df)*100:.1f}%)")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="완료된 배치 결과 다운로드 및 CSV로 병합"
    )
    parser.add_argument(
        "--output",
        help="출력 CSV 파일 경로"
    )

    args = parser.parse_args()

    # 1. 결과 다운로드
    results = download_all_completed_batches()

    if not results:
        print("ERROR: 다운로드된 결과가 없습니다.")
        return

    # 2. 파싱
    labels_df = parse_results_to_dataframe(results)

    if labels_df.empty:
        print("ERROR: 파싱된 데이터가 없습니다.")
        return

    # 3. 원본 데이터와 병합
    merged_df = merge_with_original_data(labels_df)

    if merged_df.empty:
        print("ERROR: 병합된 데이터가 없습니다.")
        return

    # 4. CSV 저장
    save_to_csv(merged_df, output_path=args.output)


if __name__ == "__main__":
    main()

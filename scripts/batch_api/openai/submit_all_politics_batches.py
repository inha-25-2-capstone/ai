#!/usr/bin/env python3
"""
OpenAI Batch API - 정치 카테고리 데이터 전체 배치 제출 (여러 배치로 나눠서)
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    import openai
except ImportError:
    print("ERROR: OpenAI 패키지가 설치되지 않았습니다.")
    exit(1)

load_dotenv()

# 프로젝트 루트 디렉토리
BASE_DIR = Path(__file__).parent.parent.parent.parent


def count_politics_data():
    """정치 카테고리 데이터 개수 확인"""
    data_path = BASE_DIR / "data" / "add_dataset.json"

    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    politics_data = [
        item for item in dataset['data']
        if item.get('doc_class', {}).get('code') == '정치'
    ]

    return len(politics_data)


def submit_all_batches(batch_size: int = 500, start_index: int = 0, delay: int = 5):
    """
    모든 정치 카테고리 데이터를 여러 배치로 나눠서 제출

    Args:
        batch_size: 한 배치당 데이터 개수
        start_index: 시작 배치 인덱스
        delay: 배치 제출 간 대기 시간 (초)
    """
    print(f"\n{'='*80}")
    print("정치 카테고리 데이터 - 전체 배치 제출")
    print(f"{'='*80}\n")

    # 전체 데이터 개수 확인
    total_data = count_politics_data()
    total_batches = (total_data + batch_size - 1) // batch_size

    print(f"전체 데이터: {total_data}개")
    print(f"배치 크기: {batch_size}개")
    print(f"필요한 배치 수: {total_batches}개")
    print(f"시작 배치 인덱스: {start_index}")
    print(f"배치 간 대기 시간: {delay}초\n")

    # API 클라이언트 초기화
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    client = openai.OpenAI(api_key=api_key)

    # 배치 정보 저장
    all_batch_info = []

    # 각 배치 제출
    for batch_index in range(start_index, total_batches):
        print(f"\n{'='*80}")
        print(f"배치 {batch_index + 1}/{total_batches} 제출 중...")
        print(f"{'='*80}\n")

        # label_politics_dataset.py 스크립트 호출
        cmd = (
            f"python scripts/batch_api/openai/label_politics_dataset.py "
            f"--mode all --batch-size {batch_size} --batch-index {batch_index}"
        )

        print(f"실행: {cmd}\n")

        result = os.system(cmd)

        if result != 0:
            print(f"\nERROR: 배치 {batch_index} 제출 실패!")
            print(f"마지막 성공한 배치: {batch_index - 1}")
            break

        print(f"\nOK: 배치 {batch_index + 1}/{total_batches} 제출 완료!")

        # 다음 배치 전 대기 (API 제한 방지)
        if batch_index < total_batches - 1:
            print(f"\n{delay}초 대기 중...")
            time.sleep(delay)

    print(f"\n{'='*80}")
    print(f"모든 배치 제출 완료!")
    print(f"{'='*80}\n")

    # 배치 정보 파일 확인
    info_file = BASE_DIR / "data" / "batch_results" / "openai" / "politics_batch_info.json"
    if info_file.exists():
        with open(info_file, 'r', encoding='utf-8') as f:
            batch_info = json.load(f)

        print(f"마지막 배치 ID: {batch_info.get('batch_id')}")
        print(f"\n상태 확인:")
        print(f"  python scripts/batch_api/openai/list_batches.py")


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI Batch API - 정치 카테고리 데이터 전체 배치 제출",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="한 배치당 데이터 개수 (기본값: 500)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="시작 배치 인덱스 (기본값: 0)"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="배치 제출 간 대기 시간 초 (기본값: 5)"
    )

    args = parser.parse_args()

    submit_all_batches(
        batch_size=args.batch_size,
        start_index=args.start_index,
        delay=args.delay
    )


if __name__ == "__main__":
    main()

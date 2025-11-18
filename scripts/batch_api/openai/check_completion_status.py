#!/usr/bin/env python3
"""
배치 완료 상태 요약
"""

import os
from dotenv import load_dotenv

try:
    import openai
except ImportError:
    print("ERROR: OpenAI 패키지가 설치되지 않았습니다.")
    exit(1)

load_dotenv()


def check_status():
    """배치 완료 상태 확인"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    client = openai.OpenAI(api_key=api_key)

    print("\n" + "="*80)
    print("배치 완료 상태 요약")
    print("="*80 + "\n")

    batches = client.batches.list(limit=100)

    # 상태별 분류
    completed = [b for b in batches.data if b.status == "completed"]
    in_progress = [b for b in batches.data if b.status in ["in_progress", "validating", "finalizing"]]
    failed = [b for b in batches.data if b.status == "failed"]

    total_needed = 21  # 배치 0~20

    print(f"필요한 총 배치: {total_needed}개")
    print(f"완료: {len(completed)}개")
    print(f"처리 중: {len(in_progress)}개")
    print(f"실패: {len(failed)}개")
    print(f"남은 배치: {total_needed - len(completed) - len(in_progress)}개")

    print(f"\n진행률: {(len(completed) + len(in_progress)) / total_needed * 100:.1f}%")

    if in_progress:
        print("\n처리 중인 배치 상세:")
        for batch in in_progress:
            if batch.request_counts and batch.request_counts.total > 0:
                completed_count = batch.request_counts.completed
                total_count = batch.request_counts.total
                progress = completed_count / total_count * 100
                print(f"  - {batch.id}: {completed_count}/{total_count} ({progress:.1f}%)")

    print("\n" + "="*80)

    # 예상 완료 시간
    if in_progress:
        print("\n재제출 프로세스가 계속 진행 중입니다.")
        print("처리 중인 배치가 완료되면 자동으로 실패한 배치를 재제출합니다.")
    else:
        print("\n모든 배치 제출이 완료되었습니다!")


if __name__ == "__main__":
    check_status()

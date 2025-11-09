#!/usr/bin/env python3
"""
Claude Batch 에러 확인 스크립트
"""

import os
import json
from pathlib import Path
from anthropic import Anthropic

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def check_batch_errors(batch_id: str):
    """배치의 에러를 자세히 확인합니다."""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        return

    client = Anthropic(api_key=api_key)

    print(f"\n{'='*60}")
    print("배치 상세 정보 확인")
    print(f"{'='*60}\n")

    # 배치 정보 가져오기
    batch = client.messages.batches.retrieve(batch_id)

    print(f"Batch ID: {batch.id}")
    print(f"처리 상태: {batch.processing_status}")
    print(f"생성 시각: {batch.created_at}")

    if hasattr(batch, 'request_counts'):
        counts = batch.request_counts
        print(f"\n요청 통계:")
        print(f"  - 총 요청: {counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired}")
        print(f"  - 성공: {counts.succeeded}")
        print(f"  - 실패: {counts.errored}")
        print(f"  - 처리 중: {counts.processing}")
        print(f"  - 취소됨: {counts.canceled}")
        print(f"  - 만료됨: {counts.expired}")

    # 에러 샘플 확인
    if batch.processing_status == "ended":
        print(f"\n{'='*60}")
        print("에러 샘플 확인 (처음 10개)")
        print(f"{'='*60}\n")

        results = client.messages.batches.results(batch_id)

        error_count = 0
        for idx, result in enumerate(results):
            if result.result.type == 'errored':
                error_count += 1
                if error_count <= 10:
                    print(f"\n[에러 #{error_count}] Custom ID: {result.custom_id}")

                    # 에러 객체 정보 출력
                    error_obj = result.result.error
                    print(f"에러 타입: {error_obj.type if hasattr(error_obj, 'type') else 'unknown'}")
                    print(f"에러 메시지: {error_obj.message if hasattr(error_obj, 'message') else 'no message'}")

                    # 전체 에러 객체 출력
                    print(f"전체 에러 정보: {error_obj}")

            if error_count >= 10:
                break

        print(f"\n총 {error_count}개 에러 발견")

if __name__ == "__main__":
    # batch_info_claude.json에서 batch_id 읽기
    base_dir = Path(__file__).parent.parent.parent.parent
    info_file = base_dir / "data" / "batch_results" / "claude" / "batch_info_claude.json"

    if info_file.exists():
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
            batch_id = info.get('batch_id')

        if batch_id:
            print(f"저장된 Batch ID 사용: {batch_id}\n")
            check_batch_errors(batch_id)
        else:
            batch_id = input("Batch ID 입력: ").strip()
            check_batch_errors(batch_id)
    else:
        batch_id = input("Batch ID 입력: ").strip()
        check_batch_errors(batch_id)

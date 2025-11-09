#!/usr/bin/env python3
"""
Claude Batch API 제출 및 관리 스크립트
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from anthropic import Anthropic

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv가 설치되지 않았습니다.")
    print("설치: pip install python-dotenv")
    print("또는 환경 변수를 직접 설정하세요.\n")


def submit_batch(client: Anthropic, input_file: str) -> str:
    """배치를 제출합니다."""

    print(f"\n{'='*60}")
    print("Claude Batch API 제출")
    print(f"{'='*60}\n")

    print(f"입력 파일: {input_file}")

    # 파일에서 요청 읽기
    with open(input_file, 'r', encoding='utf-8') as f:
        requests = [json.loads(line) for line in f]

    print(f"총 요청 수: {len(requests):,}개")

    # 배치 크기 확인
    file_size = Path(input_file).stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    print(f"파일 크기: {file_size_mb:.2f} MB")

    if len(requests) > 100000:
        print("\n⚠️  경고: 최대 요청 수(100,000개)를 초과했습니다.")
        return None

    if file_size_mb > 256:
        print("\n⚠️  경고: 최대 파일 크기(256MB)를 초과했습니다.")
        return None

    # 배치 생성
    print("\n배치 제출 중...")

    try:
        batch = client.messages.batches.create(requests=requests)

        print(f"\n✅ 배치 제출 성공!")
        print(f"\nBatch ID: {batch.id}")
        print(f"상태: {batch.processing_status}")
        print(f"생성 시각: {batch.created_at}")

        return batch.id

    except Exception as e:
        print(f"\n❌ 배치 제출 실패: {e}")
        return None


def check_batch_status(client: Anthropic, batch_id: str):
    """배치 상태를 확인합니다."""

    print(f"\n{'='*60}")
    print("배치 상태 확인")
    print(f"{'='*60}\n")

    try:
        batch = client.messages.batches.retrieve(batch_id)

        print(f"Batch ID: {batch.id}")
        print(f"처리 상태: {batch.processing_status}")
        print(f"생성 시각: {batch.created_at}")

        if hasattr(batch, 'request_counts'):
            counts = batch.request_counts
            print(f"\n요청 통계:")
            print(f"  - 처리 중: {counts.processing}")
            print(f"  - 성공: {counts.succeeded}")
            print(f"  - 실패: {counts.errored}")
            print(f"  - 취소됨: {counts.canceled}")
            print(f"  - 만료됨: {counts.expired}")

        if batch.processing_status == "ended":
            print(f"\n✅ 배치 처리 완료!")
            if hasattr(batch, 'results_url') and batch.results_url:
                print(f"결과 URL: {batch.results_url}")
        else:
            print(f"\n⏳ 배치가 아직 처리 중입니다...")

        return batch

    except Exception as e:
        print(f"\n❌ 상태 확인 실패: {e}")
        return None


def download_results(client: Anthropic, batch_id: str, output_file: str):
    """배치 결과를 다운로드합니다."""

    print(f"\n{'='*60}")
    print("배치 결과 다운로드")
    print(f"{'='*60}\n")

    try:
        # 배치 정보 가져오기
        batch = client.messages.batches.retrieve(batch_id)

        if batch.processing_status != "ended":
            print(f"⚠️  배치가 아직 완료되지 않았습니다.")
            print(f"현재 상태: {batch.processing_status}")
            return False

        # 결과 가져오기
        print(f"결과 다운로드 중...")

        results = client.messages.batches.results(batch_id)

        # JSONL 파일로 저장
        success_count = 0
        error_count = 0

        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result.model_dump(), ensure_ascii=False) + '\n')

                # 통계
                if result.result.type == 'succeeded':
                    success_count += 1
                elif result.result.type == 'errored':
                    error_count += 1

        print(f"\n✅ 결과 다운로드 완료!")
        print(f"출력 파일: {output_file}")
        print(f"\n결과 통계:")
        print(f"  - 성공: {success_count:,}개")
        print(f"  - 실패: {error_count:,}개")

        return True

    except Exception as e:
        print(f"\n❌ 결과 다운로드 실패: {e}")
        return False


def wait_for_completion(client: Anthropic, batch_id: str, check_interval: int = 60):
    """배치 완료를 기다립니다."""

    print(f"\n{'='*60}")
    print("배치 완료 대기")
    print(f"{'='*60}\n")

    print(f"Batch ID: {batch_id}")
    print(f"확인 간격: {check_interval}초\n")

    start_time = time.time()
    check_count = 0

    while True:
        check_count += 1
        elapsed = time.time() - start_time
        elapsed_min = int(elapsed / 60)

        print(f"[{check_count}회 확인] 경과 시간: {elapsed_min}분")

        try:
            batch = client.messages.batches.retrieve(batch_id)

            print(f"  상태: {batch.processing_status}")

            if hasattr(batch, 'request_counts'):
                counts = batch.request_counts
                total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
                completed = counts.succeeded + counts.errored + counts.canceled + counts.expired

                if total > 0:
                    progress = (completed / total) * 100
                    print(f"  진행률: {completed}/{total} ({progress:.1f}%)")

            if batch.processing_status == "ended":
                print(f"\n✅ 배치 처리 완료! (총 {elapsed_min}분 소요)")
                return batch

            print(f"  다음 확인까지 {check_interval}초 대기...\n")
            time.sleep(check_interval)

        except Exception as e:
            print(f"  오류: {e}")
            print(f"  {check_interval}초 후 재시도...\n")
            time.sleep(check_interval)


def save_batch_info(batch_id: str, info_file: str):
    """배치 정보를 파일에 저장합니다."""

    batch_info = {
        "batch_id": batch_id,
        "submitted_at": datetime.now().isoformat(),
        "status": "submitted"
    }

    with open(info_file, 'r', encoding='utf-8') as f:
        existing_info = json.load(f)

    existing_info.update(batch_info)

    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(existing_info, f, ensure_ascii=False, indent=2)

    print(f"배치 정보 저장: {info_file}")


def main():
    """메인 함수"""

    base_dir = Path(__file__).parent.parent

    # API 키 확인
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n❌ 오류: ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("\n설정 방법:")
        print("  Windows: set ANTHROPIC_API_KEY=your-api-key")
        print("  Linux/Mac: export ANTHROPIC_API_KEY=your-api-key")
        print()
        return

    # 파일 경로
    input_file = base_dir / "data" / "batch_claude.jsonl"
    output_file = base_dir / "data" / "batch_claude_output.jsonl"
    info_file = base_dir / "data" / "batch_info_claude.json"

    if not input_file.exists():
        print(f"\n❌ 오류: 입력 파일을 찾을 수 없습니다: {input_file}")
        print("먼저 create_claude_batch.py를 실행하세요.")
        return

    # Anthropic 클라이언트 생성
    client = Anthropic(api_key=api_key)

    # 메뉴
    print(f"\n{'='*60}")
    print("Claude Batch API 관리")
    print(f"{'='*60}\n")

    print("1. 새 배치 제출")
    print("2. 배치 상태 확인")
    print("3. 배치 제출 + 완료 대기 + 결과 다운로드 (전체 프로세스)")
    print()

    choice = input("선택 (1-3): ").strip()

    if choice == "1":
        # 배치 제출
        batch_id = submit_batch(client, str(input_file))
        if batch_id:
            save_batch_info(batch_id, str(info_file))
            print(f"\n배치 ID를 저장했습니다. 나중에 상태 확인에 사용하세요.")

    elif choice == "2":
        # 상태 확인
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
                batch_id = info.get('batch_id')

            if batch_id:
                print(f"저장된 Batch ID: {batch_id}")
                check_batch_status(client, batch_id)
            else:
                print("\n저장된 Batch ID가 없습니다.")
                batch_id = input("Batch ID 입력: ").strip()
                check_batch_status(client, batch_id)
        else:
            batch_id = input("Batch ID 입력: ").strip()
            check_batch_status(client, batch_id)

    elif choice == "3":
        # 전체 프로세스
        print("\n전체 프로세스를 시작합니다...")

        # 1. 배치 제출
        batch_id = submit_batch(client, str(input_file))
        if not batch_id:
            return

        save_batch_info(batch_id, str(info_file))

        # 2. 완료 대기
        wait_for_completion(client, batch_id, check_interval=60)

        # 3. 결과 다운로드
        download_results(client, batch_id, str(output_file))

        print(f"\n{'='*60}")
        print("전체 프로세스 완료!")
        print(f"{'='*60}\n")

        print("다음 단계:")
        print(f"  python scripts/merge_labels.py")
        print(f"  python scripts/extract_disagreements.py")
        print()

    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()

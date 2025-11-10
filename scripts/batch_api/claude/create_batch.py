#!/usr/bin/env python3
"""
Claude API Batch 처리를 위한 입력 파일 생성 스크립트
"""

import json
from pathlib import Path
from datetime import datetime


def create_claude_batch_input(input_file: str, output_file: str):
    """
    원본 데이터를 읽어 Claude API Batch 입력 파일을 생성합니다.

    Claude Message Batches API 형식:
    {
      "custom_id": "request-1",
      "params": {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [...]
      }
    }
    """

    print(f"\n{'='*60}")
    print("Claude API Batch 입력 파일 생성")
    print(f"{'='*60}\n")

    # 원본 데이터 로드
    print(f"입력 파일 로딩: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    print(f"  -> {len(articles):,}개 기사 로드됨\n")

    # Claude API Batch 요청 생성
    print("Claude API 요청 생성 중...")

    system_prompt = "당신은 정치 뉴스 분석 전문가입니다. 뉴스 기사의 논조를 정확히 분석하세요."

    batch_requests = []

    for idx, article in enumerate(articles):
        title = article.get('title', '')
        content = article.get('summary', '')

        user_prompt = f"""다음 뉴스 기사의 논조를 분석하세요.

제목: {title}

기사 내용:
{content}

위 기사의 스탠스를 다음 중 하나로 분류하세요:
0: 옹호 (해당 정책이나 인물을 긍정적으로 서술)
1: 중립 (객관적 사실 전달, 균형잡힌 시각)
2: 비판 (해당 정책이나 인물을 부정적으로 서술)

반드시 숫자(0, 1, 2) 하나만 출력하세요."""

        # Claude API Batch 요청 형식
        request = {
            "custom_id": f"article-{idx}",
            "params": {
                "model": "claude-haiku-4-5",  # Claude Haiku 4.5 모델 (빠르고 저렴)
                "max_tokens": 10,
                "temperature": 0,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            }
        }

        batch_requests.append(request)

        # 진행 상황 표시
        if (idx + 1) % 1000 == 0:
            print(f"  -> {idx + 1:,}개 요청 생성됨...")

    print(f"  -> 총 {len(batch_requests):,}개 요청 생성 완료\n")

    # JSONL 파일로 저장
    print(f"출력 파일 저장: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')

    print(f"  -> 저장 완료!")

    # 파일 크기 확인
    file_size = Path(output_file).stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    print(f"  -> 파일 크기: {file_size_mb:.2f} MB")

    # 배치 정보 저장
    batch_info = {
        "created_at": datetime.now().isoformat(),
        "input_file": str(input_file),
        "output_file": str(output_file),
        "num_articles": len(batch_requests),
        "model": "claude-haiku-4-5",
        "status": "ready_to_submit"
    }

    info_file = base_dir / "data" / "batch_results" / "claude" / "batch_info_claude.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(batch_info, f, ensure_ascii=False, indent=2)

    print(f"\n배치 정보 저장: {info_file}")

    print(f"\n{'='*60}")
    print("생성 완료!")
    print(f"{'='*60}")

    print(f"\n다음 단계:")
    print(f"  1. Claude API 콘솔에서 배치 업로드")
    print(f"     파일: {output_file}")
    print(f"  2. 배치 제출 및 처리 대기")
    print(f"  3. 결과 다운로드")
    print(f"  4. merge_labels.py 재실행하여 결과 결합")
    print()

    # Claude API 사용 팁
    print("=" * 60)
    print("[Claude Batch API 사용 방법]")
    print("=" * 60)
    print()
    print("1. Anthropic 콘솔 접속:")
    print("   https://console.anthropic.com/")
    print()
    print("2. Batches 메뉴로 이동")
    print()
    print("3. 'Create Batch' 클릭")
    print()
    print("4. 생성된 JSONL 파일 업로드:")
    print(f"   {output_file}")
    print()
    print("5. 배치 처리 완료 대기 (보통 수 시간 소요)")
    print()
    print("6. 결과 파일 다운로드:")
    print("   data/batch_claude_output.jsonl 로 저장")
    print()
    print("또는 Python SDK 사용:")
    print()
    print("  from anthropic import Anthropic")
    print("  client = Anthropic(api_key='your-api-key')")
    print()
    print("  # 파일 업로드")
    print(f"  with open('{output_file}', 'rb') as f:")
    print("      batch_file = client.files.create(")
    print("          file=f,")
    print("          purpose='batch'")
    print("      )")
    print()
    print("  # 배치 생성")
    print("  batch = client.batches.create(")
    print("      input_file_id=batch_file.id")
    print("  )")
    print()
    print("  # 상태 확인")
    print("  print(f'Batch ID: {batch.id}')")
    print("  print(f'Status: {batch.status}')")
    print()
    print("=" * 60)
    print()


def main():
    """메인 함수"""

    base_dir = Path(__file__).parent.parent.parent.parent

    # 파일 경로
    input_file = base_dir / "data" / "raw" / "bigkinds_summarized.json"
    output_file = base_dir / "data" / "batch_results" / "claude" / "batch_claude.jsonl"

    if not input_file.exists():
        print(f"오류: 입력 파일을 찾을 수 없습니다: {input_file}")
        return

    # 배치 입력 파일 생성
    create_claude_batch_input(str(input_file), str(output_file))


if __name__ == "__main__":
    main()

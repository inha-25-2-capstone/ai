"""
멀티 모델 배치 라벨링 스크립트
OpenAI Batches API + Claude Batches API를 사용하여 뉴스 기사 자동 라벨링

사용법:
    python scripts/batch_labeling.py --input data/unlabeled.csv --output data/labeled.csv

워크플로우:
    1. OpenAI GPT-4o-mini Batches로 라벨링
    2. Claude Haiku Batches로 라벨링
    3. 두 결과 비교 및 일치 여부 확인
    4. 불일치한 기사는 검토 필요 표시
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    import openai
except ImportError:
    print("❌ OpenAI 패키지가 설치되지 않았습니다.")
    print("   pip install -r requirements-scripts.txt")
    exit(1)

try:
    import anthropic
except ImportError:
    print("❌ Anthropic 패키지가 설치되지 않았습니다.")
    print("   pip install -r requirements-scripts.txt")
    exit(1)

# 환경 변수 로드
load_dotenv()


class BatchLabeler:
    """멀티 모델 배치 라벨링 클래스"""

    def __init__(self):
        # API 클라이언트 초기화
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # 라벨 정의
        self.labels = {0: "옹호 (Support)", 1: "중립 (Neutral)", 2: "비판 (Oppose)"}

    def create_openai_batch_file(self, articles, output_path="batch_openai.jsonl"):
        """OpenAI Batches API용 JSONL 파일 생성"""
        print(f"\n📝 OpenAI 배치 파일 생성 중... ({len(articles)}개 기사)")

        batch_requests = []
        for i, row in articles.iterrows():
            request = {
                "custom_id": f"article-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "당신은 정치 뉴스 분석 전문가입니다. 뉴스 기사의 논조를 정확히 분석하세요.",
                        },
                        {
                            "role": "user",
                            "content": f"""다음 뉴스 기사의 논조를 분석하세요.

제목: {row.get('title', 'N/A')}

기사 내용:
{row['text']}

위 기사의 스탠스를 다음 중 하나로 분류하세요:
0: 옹호 (해당 정책이나 인물을 긍정적으로 서술)
1: 중립 (객관적 사실 전달, 균형잡힌 시각)
2: 비판 (해당 정책이나 인물을 부정적으로 서술)

반드시 숫자(0, 1, 2) 하나만 출력하세요.""",
                        },
                    ],
                    "temperature": 0,
                    "max_tokens": 10,
                },
            }
            batch_requests.append(request)

        # JSONL 파일로 저장
        with open(output_path, "w", encoding="utf-8") as f:
            for req in batch_requests:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")

        print(f"✅ OpenAI 배치 파일 생성 완료: {output_path}")
        return output_path

    def create_claude_batch_file(self, articles, output_path="batch_claude.jsonl"):
        """Claude Batches API용 JSONL 파일 생성"""
        print(f"\n📝 Claude 배치 파일 생성 중... ({len(articles)}개 기사)")

        batch_requests = []
        for i, row in articles.iterrows():
            request = {
                "custom_id": f"article-{i}",
                "params": {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 10,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"""당신은 정치 뉴스 분석 전문가입니다.

다음 뉴스 기사의 논조를 분석하세요.

제목: {row.get('title', 'N/A')}

기사 내용:
{row['text']}

위 기사의 스탠스를 다음 중 하나로 분류하세요:
0: 옹호 (해당 정책이나 인물을 긍정적으로 서술)
1: 중립 (객관적 사실 전달, 균형잡힌 시각)
2: 비판 (해당 정책이나 인물을 부정적으로 서술)

반드시 숫자(0, 1, 2) 하나만 출력하세요.""",
                        }
                    ],
                },
            }
            batch_requests.append(request)

        # JSONL 파일로 저장
        with open(output_path, "w", encoding="utf-8") as f:
            for req in batch_requests:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")

        print(f"✅ Claude 배치 파일 생성 완료: {output_path}")
        return output_path

    def submit_openai_batch(self, batch_file):
        """OpenAI 배치 작업 제출"""
        print(f"\n📤 OpenAI 배치 업로드 중...")

        # 1. 파일 업로드
        with open(batch_file, "rb") as f:
            batch_input_file = self.openai_client.files.create(file=f, purpose="batch")

        print(f"✅ 파일 업로드 완료: {batch_input_file.id}")

        # 2. 배치 작업 생성
        batch = self.openai_client.batches.create(
            input_file_id=batch_input_file.id, endpoint="/v1/chat/completions", completion_window="24h"
        )

        print(f"✅ OpenAI 배치 작업 생성 완료")
        print(f"   배치 ID: {batch.id}")
        print(f"   상태: {batch.status}")
        print(f"   완료 예정: 24시간 이내")

        return batch.id

    def submit_claude_batch(self, batch_file):
        """Claude 배치 작업 제출"""
        print(f"\n📤 Claude 배치 업로드 중...")

        # 배치 파일 읽기
        with open(batch_file, "r", encoding="utf-8") as f:
            requests = [json.loads(line) for line in f]

        # 배치 작업 생성
        batch = self.anthropic_client.messages.batches.create(requests=requests)

        print(f"✅ Claude 배치 작업 생성 완료")
        print(f"   배치 ID: {batch.id}")
        print(f"   상태: {batch.processing_status}")
        print(f"   완료 예정: 24시간 이내")

        return batch.id

    def check_openai_batch_status(self, batch_id):
        """OpenAI 배치 작업 상태 확인"""
        batch = self.openai_client.batches.retrieve(batch_id)

        print(f"\n📊 [OpenAI] 배치 상태: {batch.status}")
        if batch.request_counts:
            print(f"   총 요청: {batch.request_counts.total}")
            print(f"   완료: {batch.request_counts.completed}")
            print(f"   실패: {batch.request_counts.failed}")

        return batch

    def check_claude_batch_status(self, batch_id):
        """Claude 배치 작업 상태 확인"""
        batch = self.anthropic_client.messages.batches.retrieve(batch_id)

        print(f"\n📊 [Claude] 배치 상태: {batch.processing_status}")
        print(f"   총 요청: {batch.request_counts.processing + batch.request_counts.succeeded + batch.request_counts.errored}")
        print(f"   완료: {batch.request_counts.succeeded}")
        print(f"   실패: {batch.request_counts.errored}")

        return batch

    def download_openai_results(self, batch_id):
        """OpenAI 배치 결과 다운로드"""
        print(f"\n⬇️ OpenAI 결과 다운로드 중...")

        batch = self.openai_client.batches.retrieve(batch_id)

        if batch.status != "completed":
            print(f"⚠️ 아직 처리 중입니다: {batch.status}")
            return None

        # 결과 파일 다운로드
        result_file_id = batch.output_file_id
        result = self.openai_client.files.content(result_file_id)

        # 결과 파싱
        results = {}
        for line in result.text.strip().split("\n"):
            data = json.loads(line)
            custom_id = data["custom_id"]
            article_idx = int(custom_id.split("-")[1])

            try:
                label = data["response"]["body"]["choices"][0]["message"]["content"].strip()
                results[article_idx] = int(label)
            except Exception as e:
                print(f"⚠️ 파싱 오류 ({custom_id}): {e}")
                results[article_idx] = None

        print(f"✅ OpenAI 결과 다운로드 완료: {len(results)}개")
        return results

    def download_claude_results(self, batch_id):
        """Claude 배치 결과 다운로드"""
        print(f"\n⬇️ Claude 결과 다운로드 중...")

        batch = self.anthropic_client.messages.batches.retrieve(batch_id)

        if batch.processing_status != "ended":
            print(f"⚠️ 아직 처리 중입니다: {batch.processing_status}")
            return None

        # 결과 가져오기
        results = {}
        for result in self.anthropic_client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            article_idx = int(custom_id.split("-")[1])

            try:
                if result.result.type == "succeeded":
                    label = result.result.message.content[0].text.strip()
                    results[article_idx] = int(label)
                else:
                    print(f"⚠️ 실패 ({custom_id}): {result.result.type}")
                    results[article_idx] = None
            except Exception as e:
                print(f"⚠️ 파싱 오류 ({custom_id}): {e}")
                results[article_idx] = None

        print(f"✅ Claude 결과 다운로드 완료: {len(results)}개")
        return results

    def compare_results(self, openai_results, claude_results):
        """두 모델의 결과 비교"""
        print(f"\n🔍 결과 비교 중...")

        comparison = {}
        agreements = 0
        disagreements = 0

        for idx in openai_results.keys():
            openai_label = openai_results.get(idx)
            claude_label = claude_results.get(idx)

            if openai_label is None or claude_label is None:
                comparison[idx] = {
                    "openai": openai_label,
                    "claude": claude_label,
                    "agreement": False,
                    "final_label": None,
                    "needs_review": True,
                    "reason": "API 오류",
                }
                disagreements += 1
            elif openai_label == claude_label:
                comparison[idx] = {
                    "openai": openai_label,
                    "claude": claude_label,
                    "agreement": True,
                    "final_label": openai_label,
                    "needs_review": False,
                    "reason": "일치",
                }
                agreements += 1
            else:
                comparison[idx] = {
                    "openai": openai_label,
                    "claude": claude_label,
                    "agreement": False,
                    "final_label": None,
                    "needs_review": True,
                    "reason": "불일치",
                }
                disagreements += 1

        print(f"\n📊 비교 결과:")
        print(f"   ✅ 일치: {agreements}개 ({agreements / len(comparison) * 100:.1f}%)")
        print(f"   ⚠️ 불일치: {disagreements}개 ({disagreements / len(comparison) * 100:.1f}%)")

        return comparison

    def save_results(self, df, comparison, output_path):
        """결과를 CSV로 저장"""
        print(f"\n💾 결과 저장 중...")

        # 비교 결과를 데이터프레임에 추가
        df["label_openai"] = df.index.map(lambda i: comparison.get(i, {}).get("openai"))
        df["label_claude"] = df.index.map(lambda i: comparison.get(i, {}).get("claude"))
        df["label_final"] = df.index.map(lambda i: comparison.get(i, {}).get("final_label"))
        df["agreement"] = df.index.map(lambda i: comparison.get(i, {}).get("agreement", False))
        df["needs_review"] = df.index.map(lambda i: comparison.get(i, {}).get("needs_review", True))
        df["review_reason"] = df.index.map(lambda i: comparison.get(i, {}).get("reason", ""))

        # 저장
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ 결과 저장 완료: {output_path}")

        # 검토 필요 항목 별도 저장
        review_df = df[df["needs_review"]]
        if len(review_df) > 0:
            review_path = output_path.replace(".csv", "_review_needed.csv")
            review_df.to_csv(review_path, index=False, encoding="utf-8-sig")
            print(f"⚠️ 검토 필요 항목 저장: {review_path} ({len(review_df)}개)")


def main():
    parser = argparse.ArgumentParser(description="멀티 모델 배치 라벨링 스크립트")
    parser.add_argument("--input", required=True, help="입력 CSV 파일 경로")
    parser.add_argument("--output", default="data/batch_labeled.csv", help="출력 CSV 파일 경로")
    parser.add_argument("--mode", choices=["submit", "check", "download"], default="submit", help="실행 모드")
    parser.add_argument("--openai-batch-id", help="OpenAI 배치 ID (check/download 모드)")
    parser.add_argument("--claude-batch-id", help="Claude 배치 ID (check/download 모드)")
    args = parser.parse_args()

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY=your-key 추가하세요.")
        return

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   .env 파일에 ANTHROPIC_API_KEY=your-key 추가하세요.")
        return

    labeler = BatchLabeler()

    if args.mode == "submit":
        # 데이터 로드
        print(f"\n📂 데이터 로드 중: {args.input}")
        df = pd.read_csv(args.input)
        print(f"✅ {len(df)}개 기사 로드 완료")

        # 배치 파일 생성
        openai_file = labeler.create_openai_batch_file(df)
        claude_file = labeler.create_claude_batch_file(df)

        # 배치 제출
        openai_batch_id = labeler.submit_openai_batch(openai_file)
        claude_batch_id = labeler.submit_claude_batch(claude_file)

        # 배치 ID 저장
        batch_info = {
            "openai_batch_id": openai_batch_id,
            "claude_batch_id": claude_batch_id,
            "submitted_at": datetime.now().isoformat(),
            "num_articles": len(df),
        }

        with open("batch_info.json", "w", encoding="utf-8") as f:
            json.dump(batch_info, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 배치 제출 완료!")
        print(f"\n💡 다음 단계:")
        print(f"   1. 24시간 후 상태 확인:")
        print(
            f"      python scripts/batch_labeling.py --mode check --openai-batch-id {openai_batch_id} --claude-batch-id {claude_batch_id}"
        )
        print(f"\n   2. 완료 후 결과 다운로드:")
        print(
            f"      python scripts/batch_labeling.py --mode download --input {args.input} --output {args.output} --openai-batch-id {openai_batch_id} --claude-batch-id {claude_batch_id}"
        )

    elif args.mode == "check":
        # 상태 확인
        if args.openai_batch_id:
            labeler.check_openai_batch_status(args.openai_batch_id)

        if args.claude_batch_id:
            labeler.check_claude_batch_status(args.claude_batch_id)

    elif args.mode == "download":
        # 결과 다운로드
        if not args.openai_batch_id or not args.claude_batch_id:
            print("❌ --openai-batch-id와 --claude-batch-id가 필요합니다.")
            return

        # 원본 데이터 로드
        df = pd.read_csv(args.input)

        # 결과 다운로드
        openai_results = labeler.download_openai_results(args.openai_batch_id)
        claude_results = labeler.download_claude_results(args.claude_batch_id)

        if openai_results is None or claude_results is None:
            print("❌ 아직 배치 작업이 완료되지 않았습니다.")
            return

        # 결과 비교
        comparison = labeler.compare_results(openai_results, claude_results)

        # 결과 저장
        labeler.save_results(df, comparison, args.output)

        print(f"\n🎉 라벨링 완료!")


if __name__ == "__main__":
    main()

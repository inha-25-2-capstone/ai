"""
OpenAI Batches API 자동 라벨링 스크립트
GPT-4o-mini Batches API를 사용하여 뉴스 기사 자동 라벨링 (50% 할인)

사용법:
    # 1단계: 배치 제출
    python scripts/batch_labeling_openai.py --input data/unlabeled.csv --mode submit

    # 2단계: 상태 확인
    python scripts/batch_labeling_openai.py --batch-id batch_xxx --mode check

    # 3단계: 결과 다운로드
    python scripts/batch_labeling_openai.py --batch-id batch_xxx --input data/unlabeled.csv --output data/labeled.csv --mode download
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

# 환경 변수 로드
load_dotenv()


class OpenAIBatchLabeler:
    """OpenAI Batches API 라벨링 클래스"""

    def __init__(self):
        # API 클라이언트 초기화
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

        self.client = openai.OpenAI(api_key=api_key)

        # 라벨 정의
        self.labels = {
            0: "옹호 (Support)",
            1: "중립 (Neutral)",
            2: "비판 (Oppose)"
        }

    def load_data(self, file_path):
        """CSV, JSON, JSONL 파일 자동 감지 및 로드"""
        file_ext = Path(file_path).suffix.lower()

        print(f"\n📂 데이터 로드 중: {file_path}")

        if file_ext == ".csv":
            df = pd.read_csv(file_path)
            print(f"✅ CSV 파일 로드 완료: {len(df)}개 기사")
        elif file_ext == ".json":
            df = pd.read_json(file_path)
            print(f"✅ JSON 파일 로드 완료: {len(df)}개 기사")
        elif file_ext == ".jsonl":
            df = pd.read_json(file_path, lines=True)
            print(f"✅ JSONL 파일 로드 완료: {len(df)}개 기사")
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_ext}\n   지원 형식: .csv, .json, .jsonl")

        # summary를 text로 사용 (기본 데이터셋 구조: title, summary, topic)
        if "summary" in df.columns and "text" not in df.columns:
            print("💡 'summary' 필드를 텍스트로 사용합니다.")
            df["text"] = df["summary"]
        elif "text" not in df.columns and "summary" not in df.columns:
            raise ValueError("'text' 또는 'summary' 컬럼이 필요합니다.")

        return df

    def create_batch_file(self, articles, output_path="batch_openai.jsonl"):
        """Batches API용 JSONL 파일 생성"""
        print(f"\n📝 배치 파일 생성 중... ({len(articles)}개 기사)")

        batch_requests = []
        for i, row in articles.iterrows():
            request = {
                "custom_id": f"article-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",  # 50% 할인 적용
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
                    "temperature": 0,  # 일관성을 위해 0
                    "max_tokens": 10,
                },
            }
            batch_requests.append(request)

        # JSONL 파일로 저장
        with open(output_path, "w", encoding="utf-8") as f:
            for req in batch_requests:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")

        print(f"✅ 배치 파일 생성 완료: {output_path}")
        return output_path

    def submit_batch(self, batch_file):
        """배치 작업 제출"""
        print(f"\n📤 배치 업로드 중...")

        # 1. 파일 업로드
        with open(batch_file, "rb") as f:
            batch_input_file = self.client.files.create(
                file=f,
                purpose="batch"
            )

        print(f"✅ 파일 업로드 완료: {batch_input_file.id}")

        # 2. 배치 작업 생성
        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        print(f"\n✅ 배치 작업 생성 완료!")
        print(f"   배치 ID: {batch.id}")
        print(f"   상태: {batch.status}")
        print(f"   완료 예정: 24시간 이내")

        return batch.id

    def check_status(self, batch_id):
        """배치 작업 상태 확인"""
        batch = self.client.batches.retrieve(batch_id)

        print(f"\n📊 배치 상태")
        print(f"   ID: {batch.id}")
        print(f"   상태: {batch.status}")

        if batch.request_counts:
            total = batch.request_counts.total
            completed = batch.request_counts.completed
            failed = batch.request_counts.failed

            print(f"   진행률: {completed}/{total} ({completed/total*100:.1f}%)")
            if failed > 0:
                print(f"   ⚠️ 실패: {failed}개")

        if batch.status == "completed":
            print(f"   ✅ 완료! 결과 다운로드 가능")
        elif batch.status == "failed":
            print(f"   ❌ 실패: {batch.errors}")
        else:
            print(f"   ⏳ 처리 중... 나중에 다시 확인하세요")

        return batch

    def download_results(self, batch_id):
        """배치 결과 다운로드"""
        print(f"\n⬇️ 결과 다운로드 중...")

        batch = self.client.batches.retrieve(batch_id)

        if batch.status != "completed":
            print(f"⚠️ 아직 처리 중입니다: {batch.status}")
            return None

        # 결과 파일 다운로드
        result_file_id = batch.output_file_id
        result = self.client.files.content(result_file_id)

        # 결과 파싱
        results = {}
        errors = []

        for line in result.text.strip().split("\n"):
            data = json.loads(line)
            custom_id = data["custom_id"]
            article_idx = int(custom_id.split("-")[1])

            try:
                # 응답에서 라벨 추출
                label_str = data["response"]["body"]["choices"][0]["message"]["content"].strip()

                # 숫자로 변환
                label = int(label_str)

                # 유효성 검사
                if label not in [0, 1, 2]:
                    print(f"⚠️ 잘못된 라벨 ({custom_id}): {label}")
                    results[article_idx] = None
                    errors.append({
                        'article_idx': article_idx,
                        'error': f'Invalid label: {label}'
                    })
                else:
                    results[article_idx] = label

            except (KeyError, ValueError, IndexError) as e:
                print(f"⚠️ 파싱 오류 ({custom_id}): {e}")
                results[article_idx] = None
                errors.append({
                    'article_idx': article_idx,
                    'error': str(e)
                })

        print(f"✅ 결과 다운로드 완료")
        print(f"   성공: {sum(1 for v in results.values() if v is not None)}개")
        print(f"   실패: {len(errors)}개")

        return results, errors

    def save_results(self, df, results, output_path):
        """결과를 CSV 또는 JSON으로 저장"""
        print(f"\n💾 결과 저장 중...")

        # 결과를 데이터프레임에 추가
        df["label"] = df.index.map(lambda i: results.get(i))
        df["label_name"] = df["label"].map(lambda x: self.labels.get(x, "오류"))
        df["labeled"] = df["label"].notna()
        df["labeled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 파일 형식 확인
        file_ext = Path(output_path).suffix.lower()

        # 저장
        if file_ext == ".csv":
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"✅ 결과 저장 완료 (CSV): {output_path}")
        elif file_ext == ".json":
            df.to_json(output_path, orient="records", force_ascii=False, indent=2)
            print(f"✅ 결과 저장 완료 (JSON): {output_path}")
        elif file_ext == ".jsonl":
            df.to_json(output_path, orient="records", force_ascii=False, lines=True)
            print(f"✅ 결과 저장 완료 (JSONL): {output_path}")
        else:
            # 기본값: CSV
            output_path = output_path + ".csv"
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"✅ 결과 저장 완료 (CSV): {output_path}")

        # 통계
        label_counts = df["label"].value_counts().sort_index()
        print(f"\n📊 라벨 분포:")
        for label, count in label_counts.items():
            if pd.notna(label):
                label_int = int(label)
                print(f"   {label_int} ({self.labels[label_int]}): {count}개 ({count/len(df)*100:.1f}%)")

        # 실패한 항목
        failed = df[df["label"].isna()]
        if len(failed) > 0:
            # 실패 파일도 같은 형식으로 저장
            if file_ext == ".csv":
                failed_path = output_path.replace(".csv", "_failed.csv")
                failed.to_csv(failed_path, index=False, encoding="utf-8-sig")
            elif file_ext == ".json":
                failed_path = output_path.replace(".json", "_failed.json")
                failed.to_json(failed_path, orient="records", force_ascii=False, indent=2)
            elif file_ext == ".jsonl":
                failed_path = output_path.replace(".jsonl", "_failed.jsonl")
                failed.to_json(failed_path, orient="records", force_ascii=False, lines=True)
            else:
                failed_path = output_path.replace(".csv", "_failed.csv")
                failed.to_csv(failed_path, index=False, encoding="utf-8-sig")

            print(f"\n⚠️ 라벨링 실패 항목: {failed_path} ({len(failed)}개)")


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI Batches API 자동 라벨링",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 배치 제출 (CSV)
  python scripts/batch_labeling_openai.py --input data/unlabeled.csv --mode submit

  # 배치 제출 (JSON)
  python scripts/batch_labeling_openai.py --input data/unlabeled.json --mode submit

  # 배치 제출 (JSONL)
  python scripts/batch_labeling_openai.py --input data/unlabeled.jsonl --mode submit

  # 상태 확인
  python scripts/batch_labeling_openai.py --batch-id batch_xxx --mode check

  # 결과 다운로드 (CSV)
  python scripts/batch_labeling_openai.py --batch-id batch_xxx --input data/unlabeled.csv --output data/labeled.csv --mode download

  # 결과 다운로드 (JSON)
  python scripts/batch_labeling_openai.py --batch-id batch_xxx --input data/unlabeled.json --output data/labeled.json --mode download
        """
    )
    parser.add_argument("--input", help="입력 파일 경로 (.csv, .json, .jsonl)")
    parser.add_argument("--output", default="data/labeled_openai.csv", help="출력 파일 경로 (.csv, .json, .jsonl)")
    parser.add_argument(
        "--mode",
        choices=["submit", "check", "download"],
        required=True,
        help="실행 모드"
    )
    parser.add_argument("--batch-id", help="배치 ID (check/download 모드)")
    args = parser.parse_args()

    try:
        labeler = OpenAIBatchLabeler()
    except ValueError as e:
        print(f"❌ {e}")
        print("   .env 파일에 OPENAI_API_KEY를 설정하세요.")
        return

    if args.mode == "submit":
        if not args.input:
            print("❌ --input 파일이 필요합니다.")
            return

        # 데이터 로드 (CSV, JSON, JSONL 자동 감지)
        try:
            df = labeler.load_data(args.input)
        except ValueError as e:
            print(f"❌ {e}")
            return

        # 배치 파일 생성
        batch_file = labeler.create_batch_file(df)

        # 배치 제출
        batch_id = labeler.submit_batch(batch_file)

        # 배치 ID 저장
        batch_info = {
            "batch_id": batch_id,
            "submitted_at": datetime.now().isoformat(),
            "input_file": args.input,
            "num_articles": len(df),
        }

        info_file = "batch_info_openai.json"
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(batch_info, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 배치 정보 저장: {info_file}")

        print(f"\n💡 다음 단계:")
        print(f"\n1. 상태 확인 (몇 시간 후):")
        print(f"   python scripts/batch_labeling_openai.py --mode check --batch-id {batch_id}")
        print(f"\n2. 결과 다운로드 (완료 후):")
        print(f"   python scripts/batch_labeling_openai.py --mode download --batch-id {batch_id} --input {args.input} --output {args.output}")

    elif args.mode == "check":
        if not args.batch_id:
            print("❌ --batch-id가 필요합니다.")
            return

        # 상태 확인
        labeler.check_status(args.batch_id)

    elif args.mode == "download":
        if not args.batch_id:
            print("❌ --batch-id가 필요합니다.")
            return

        if not args.input:
            print("❌ --input 파일이 필요합니다.")
            return

        # 원본 데이터 로드 (CSV, JSON, JSONL 자동 감지)
        try:
            df = labeler.load_data(args.input)
        except ValueError as e:
            print(f"❌ {e}")
            return

        # 결과 다운로드
        results, errors = labeler.download_results(args.batch_id)

        if results is None:
            print("❌ 아직 배치 작업이 완료되지 않았습니다.")
            return

        # 결과 저장
        labeler.save_results(df, results, args.output)

        print(f"\n🎉 라벨링 완료!")


if __name__ == "__main__":
    main()

"""
LLM 기반 자동 라벨링 스크립트 (ChatGPT + Gemini)

사용법:
    # ChatGPT로 라벨링
    python scripts/auto_labeling.py --input data/unlabeled.csv --output data/labeled.csv --provider openai --api-key YOUR_KEY

    # Gemini로 라벨링
    python scripts/auto_labeling.py --input data/unlabeled.csv --output data/labeled.csv --provider gemini --api-key YOUR_KEY

    # 두 API 병행 사용 (교차 검증)
    python scripts/auto_labeling.py --input data/unlabeled.csv --output data/labeled.csv --provider both --openai-key KEY1 --gemini-key KEY2
"""

import argparse
import pandas as pd
import json
import time
import os
from typing import Dict, List, Optional
from tqdm.auto import tqdm


# ============================================================================
# 프롬프트 템플릿
# ============================================================================

STANCE_LABELING_PROMPT = """당신은 뉴스 기사의 논조를 분석하는 전문가입니다.

다음 뉴스 기사를 읽고, 기사의 논조(스탠스)를 다음 3가지 중 하나로 분류하세요:

**0: 옹호 (Support)**
- 특정 이슈, 정책, 인물에 대해 긍정적이고 지지하는 논조
- 긍정적 표현 사용, 이점과 기대 효과 강조
- 예시: "이번 정책은 경제 성장에 큰 도움이 될 것으로 기대된다."

**1: 중립 (Neutral)**
- 특정 입장을 취하지 않고 사실만을 전달하는 중립적 논조
- 감정적 표현 없이 객관적 서술, 양측 의견 균형있게 제시
- 예시: "정부가 새로운 정책을 발표했다. 시행 시기는 내년 초로 예정되어 있다."

**2: 비판 (Oppose)**
- 특정 이슈, 정책, 인물에 대해 부정적이고 비판하는 논조
- 부정적 표현 사용, 문제점과 우려사항 강조
- 예시: "정부의 정책은 현실을 제대로 반영하지 못한 채 졸속으로 추진되고 있다."

---

**뉴스 기사:**
{text}

---

**중요:**
1. 기사 전체의 논조를 파악하세요 (제목만 보지 말고 전체 맥락 고려)
2. 기자의 의견 vs 인용문을 구분하세요
3. 애매한 경우 중립(1)로 분류하세요
4. 반드시 0, 1, 2 중 하나의 숫자만 답변하세요

**답변 형식 (JSON):**
{{
  "label": 0 또는 1 또는 2,
  "confidence": 0.0~1.0 (확신도),
  "reason": "판단 근거를 한 문장으로"
}}
"""


# ============================================================================
# API 클라이언트
# ============================================================================

class LabelingClient:
    """LLM 기반 라벨링 클라이언트 (추상 클래스)"""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.cost = 0.0
        self.request_count = 0

    def label_text(self, text: str) -> Dict:
        """텍스트 라벨링 (서브클래스에서 구현)"""
        raise NotImplementedError

    def _parse_response(self, response_text: str) -> Dict:
        """응답 파싱"""
        try:
            # JSON 파싱 시도
            result = json.loads(response_text)

            # 유효성 검증
            if "label" not in result:
                return {"error": "label 필드 누락"}

            label = int(result["label"])
            if label not in [0, 1, 2]:
                return {"error": f"잘못된 레이블 값: {label}"}

            return {
                "label": label,
                "confidence": float(result.get("confidence", 0.5)),
                "reason": result.get("reason", "")
            }
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 숫자만 추출 시도
            import re
            numbers = re.findall(r'\b[012]\b', response_text)
            if numbers:
                return {
                    "label": int(numbers[0]),
                    "confidence": 0.5,
                    "reason": "JSON 파싱 실패, 숫자만 추출"
                }
            return {"error": f"응답 파싱 실패: {response_text[:100]}"}


class OpenAIClient(LabelingClient):
    """OpenAI (ChatGPT) 클라이언트"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        super().__init__(api_key, model)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai 패키지가 필요합니다: pip install openai")

    def label_text(self, text: str) -> Dict:
        """ChatGPT로 라벨링"""
        try:
            prompt = STANCE_LABELING_PROMPT.format(text=text)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 뉴스 논조 분석 전문가입니다. 항상 JSON 형식으로 답변하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )

            self.request_count += 1

            # 비용 계산 (gpt-4o-mini 기준: input $0.15/1M tokens, output $0.6/1M tokens)
            if self.model == "gpt-4o-mini":
                input_cost = response.usage.prompt_tokens / 1_000_000 * 0.15
                output_cost = response.usage.completion_tokens / 1_000_000 * 0.6
                self.cost += input_cost + output_cost

            response_text = response.choices[0].message.content
            result = self._parse_response(response_text)
            result["provider"] = "openai"
            result["model"] = self.model

            return result

        except Exception as e:
            return {"error": f"OpenAI API 오류: {str(e)}"}


class GeminiClient(LabelingClient):
    """Google Gemini 클라이언트"""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        super().__init__(api_key, model)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("google-generativeai 패키지가 필요합니다: pip install google-generativeai")

    def label_text(self, text: str) -> Dict:
        """Gemini로 라벨링"""
        try:
            import google.generativeai as genai
            prompt = STANCE_LABELING_PROMPT.format(text=text)

            # 안전 필터 설정 (정치 뉴스 분석을 위해 완화)
            safety_settings = {
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }

            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 200,
                },
                safety_settings=safety_settings
            )

            self.request_count += 1

            # 비용 계산 (Gemini 1.5 Flash: input $0.075/1M tokens, output $0.3/1M tokens)
            # 참고: 정확한 토큰 수는 response.usage_metadata에서 가져올 수 있음
            if hasattr(response, 'usage_metadata'):
                input_cost = response.usage_metadata.prompt_token_count / 1_000_000 * 0.075
                output_cost = response.usage_metadata.candidates_token_count / 1_000_000 * 0.3
                self.cost += input_cost + output_cost

            response_text = response.text
            result = self._parse_response(response_text)
            result["provider"] = "gemini"
            result["model"] = self.model

            return result

        except Exception as e:
            return {"error": f"Gemini API 오류: {str(e)}"}


# ============================================================================
# 자동 라벨링 메인 로직
# ============================================================================

class AutoLabeler:
    """자동 라벨링 클래스"""

    def __init__(self, clients: List[LabelingClient], save_interval: int = 50):
        self.clients = clients
        self.save_interval = save_interval
        self.results = []

    def label_dataset(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """데이터셋 전체 라벨링"""
        print(f"\n{'='*60}")
        print(f"🤖 자동 라벨링 시작")
        print(f"{'='*60}")
        print(f"전체 샘플 수: {len(df)}개")
        print(f"사용 API: {[c.__class__.__name__ for c in self.clients]}")
        print(f"{'='*60}\n")

        labeled_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="라벨링 중"):
            text = row[text_column]

            # 여러 클라이언트로 라벨링 (교차 검증용)
            labels = []
            confidences = []
            reasons = []
            providers = []

            for client in self.clients:
                result = client.label_text(text)

                if "error" in result:
                    print(f"\n⚠️  오류 발생 (행 {idx}): {result['error']}")
                    continue

                labels.append(result["label"])
                confidences.append(result.get("confidence", 0.5))
                reasons.append(result.get("reason", ""))
                providers.append(result.get("provider", "unknown"))

                # API Rate Limit 방지
                time.sleep(0.5)

            if not labels:
                print(f"\n❌ 모든 API 실패 (행 {idx})")
                continue

            # 라벨 집계
            if len(labels) == 1:
                final_label = labels[0]
                final_confidence = confidences[0]
                agreement = True
            else:
                # 여러 API 사용 시: 과반수 투표
                from collections import Counter
                label_counts = Counter(labels)
                final_label = label_counts.most_common(1)[0][0]
                final_confidence = sum(confidences) / len(confidences)
                agreement = all(l == final_label for l in labels)

            labeled_data.append({
                **row.to_dict(),
                "label": final_label,
                "confidence": round(final_confidence, 4),
                "agreement": agreement,
                "providers": ",".join(providers),
                "reasons": " | ".join(reasons) if reasons else ""
            })

            # 중간 저장
            if len(labeled_data) % self.save_interval == 0:
                self._save_checkpoint(labeled_data, f"checkpoint_{len(labeled_data)}.csv")

        result_df = pd.DataFrame(labeled_data)

        # 통계 출력
        self._print_statistics(result_df)

        return result_df

    def _save_checkpoint(self, data: List[Dict], filename: str):
        """중간 저장"""
        checkpoint_dir = "data/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        filepath = os.path.join(checkpoint_dir, filename)
        pd.DataFrame(data).to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n💾 체크포인트 저장: {filepath}")

    def _print_statistics(self, df: pd.DataFrame):
        """통계 출력"""
        print(f"\n\n{'='*60}")
        print(f"📊 라벨링 완료 통계")
        print(f"{'='*60}")

        print(f"\n전체 샘플 수: {len(df)}개")

        # 레이블 분포
        label_names = {0: '옹호', 1: '중립', 2: '비판'}
        print(f"\n레이블 분포:")
        for label in [0, 1, 2]:
            count = (df['label'] == label).sum()
            percentage = count / len(df) * 100 if len(df) > 0 else 0
            print(f"  {label} ({label_names[label]}): {count}개 ({percentage:.1f}%)")

        # 신뢰도 통계
        if 'confidence' in df.columns:
            print(f"\n신뢰도 통계:")
            print(f"  평균: {df['confidence'].mean():.3f}")
            print(f"  중간값: {df['confidence'].median():.3f}")
            print(f"  최소: {df['confidence'].min():.3f}")
            print(f"  최대: {df['confidence'].max():.3f}")

        # 일치도 (여러 API 사용 시)
        if 'agreement' in df.columns:
            agreement_rate = df['agreement'].sum() / len(df) * 100
            print(f"\nAPI 간 일치도: {agreement_rate:.1f}%")

        # 비용 통계
        print(f"\nAPI 사용 통계:")
        for client in self.clients:
            print(f"  {client.__class__.__name__}:")
            print(f"    요청 수: {client.request_count}회")
            print(f"    예상 비용: ${client.cost:.4f}")

        total_cost = sum(c.cost for c in self.clients)
        print(f"\n총 예상 비용: ${total_cost:.4f}")

        print(f"{'='*60}\n")


# ============================================================================
# 메인 함수
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LLM 기반 자동 라벨링')
    parser.add_argument('--input', type=str, required=True, help='입력 CSV 파일')
    parser.add_argument('--output', type=str, required=True, help='출력 CSV 파일')
    parser.add_argument('--text-column', type=str, default='text', help='텍스트 컬럼 이름')
    parser.add_argument('--provider', type=str, choices=['openai', 'gemini', 'both'], default='openai',
                        help='사용할 API: openai, gemini, both')
    parser.add_argument('--openai-key', type=str, help='OpenAI API 키')
    parser.add_argument('--gemini-key', type=str, help='Gemini API 키')
    parser.add_argument('--openai-model', type=str, default='gpt-4o-mini', help='OpenAI 모델')
    parser.add_argument('--gemini-model', type=str, default='gemini-1.5-flash', help='Gemini 모델')
    parser.add_argument('--save-interval', type=int, default=50, help='중간 저장 간격')
    parser.add_argument('--sample', type=int, help='테스트용 샘플 개수')

    args = parser.parse_args()

    # 데이터 로드
    print(f"\n[INFO] Loading data: {args.input}")
    df = pd.read_csv(args.input)

    if args.sample:
        print(f"⚠️  테스트 모드: 첫 {args.sample}개만 라벨링")
        df = df.head(args.sample)

    print(f"전체 샘플 수: {len(df)}개\n")

    # 클라이언트 초기화
    clients = []

    if args.provider in ['openai', 'both']:
        if not args.openai_key:
            print("❌ --openai-key 필수")
            return
        print(f"✅ OpenAI 클라이언트 초기화 (모델: {args.openai_model})")
        clients.append(OpenAIClient(args.openai_key, args.openai_model))

    if args.provider in ['gemini', 'both']:
        if not args.gemini_key:
            print("❌ --gemini-key 필수")
            return
        print(f"✅ Gemini 클라이언트 초기화 (모델: {args.gemini_model})")
        clients.append(GeminiClient(args.gemini_key, args.gemini_model))

    # 자동 라벨링
    labeler = AutoLabeler(clients, save_interval=args.save_interval)
    result_df = labeler.label_dataset(df, text_column=args.text_column)

    # 결과 저장
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n✅ 라벨링 완료: {args.output}")

    print(f"\n다음 단계:")
    print(f"1. 신뢰도가 낮은 샘플 검수 (confidence < 0.7)")
    print(f"2. API 간 불일치 샘플 검수 (agreement = False)")
    print(f"3. 검증 완료 후 모델 학습")


if __name__ == '__main__':
    main()

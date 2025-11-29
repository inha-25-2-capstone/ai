"""
배포 서버 기사 스탠스 분석 스크립트

배포된 백엔드 서버에서 기사를 가져와 스탠스 분석 후 결과를 저장합니다.

Usage:
    # 기본 실행 (분석 안 된 기사 가져와서 분석)
    python scripts/analyze_server_articles.py

    # 특정 토픽의 기사 분석
    python scripts/analyze_server_articles.py --topic_id 1

    # 특정 날짜 기사 분석
    python scripts/analyze_server_articles.py --date 2024-11-26

    # 결과를 서버에 저장 (API 준비 후)
    python scripts/analyze_server_articles.py --save_to_server
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, date

warnings.filterwarnings("ignore")

import requests
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer
import pandas as pd
from tqdm import tqdm

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 서버 설정
API_BASE_URL = "https://politics-news-api.onrender.com"


class StanceClassifier(nn.Module):
    """KoBERT 기반 스탠스 분류 모델"""

    def __init__(self, n_classes=3, dropout=0.3, model_name="skt/kobert-base-v1"):
        super(StanceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class StancePredictor:
    """스탠스 예측기"""

    def __init__(self, model_path, model_name="skt/kobert-base-v1", max_length=512, device=None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 라벨 정의
        self.label_names = ["옹호", "중립", "비판"]
        self.label_names_en = ["support", "neutral", "oppose"]

        # 토크나이저 로드 (학습 시 사용한 monologg/kobert 사용)
        print("토크나이저 로딩 중... (monologg/kobert)")
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

        # 모델 로드
        print(f"모델 로딩 중... ({model_path})")
        self.model = StanceClassifier(n_classes=3, dropout=0.3, model_name=model_name)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[OK] 모델 로드 완료! (Device: {self.device})")

    def predict_batch(self, texts, batch_size=16, show_progress=True):
        """여러 텍스트의 스탠스 배치 예측"""
        results = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="스탠스 분석 중")

        for i in iterator:
            batch_texts = texts[i : i + batch_size]

            # 토크나이징
            encodings = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            # 예측
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probs, dim=1)
                confidences = torch.max(probs, dim=1).values

            # 결과 저장
            for j, text in enumerate(batch_texts):
                pred_class = predicted_classes[j].item()
                results.append(
                    {
                        "predicted_label": pred_class,
                        "predicted_stance": self.label_names_en[pred_class],
                        "predicted_stance_kr": self.label_names[pred_class],
                        "confidence": round(confidences[j].item(), 4),
                        "prob_support": round(probs[j][0].item(), 4),
                        "prob_neutral": round(probs[j][1].item(), 4),
                        "prob_oppose": round(probs[j][2].item(), 4),
                    }
                )

        return results


class ArticleFetcher:
    """배포 서버에서 기사를 가져오는 클래스"""

    def __init__(self, base_url=API_BASE_URL):
        self.base_url = base_url

    def fetch_articles(self, topic_id=None, date_str=None, stance=None, limit=100, page=1):
        """
        기사 목록 가져오기

        Args:
            topic_id: 특정 토픽 ID
            date_str: 날짜 필터 (YYYY-MM-DD)
            stance: 스탠스 필터 (support/neutral/oppose/None=미분석)
            limit: 한 번에 가져올 기사 수
            page: 페이지 번호
        """
        params = {"limit": limit, "page": page}

        if topic_id:
            params["topic_id"] = topic_id
        if date_str:
            params["start_date"] = date_str
            params["end_date"] = date_str
        if stance:
            params["stance"] = stance

        try:
            response = requests.get(f"{self.base_url}/api/articles", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("data", data.get("items", data.get("articles", [])))
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] 기사 가져오기 실패: {e}")
            return []

    def fetch_article_detail(self, article_id):
        """기사 상세 정보 가져오기"""
        try:
            response = requests.get(f"{self.base_url}/api/articles/{article_id}", timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] 기사 {article_id} 상세 정보 가져오기 실패: {e}")
            return None

    def fetch_topics(self, date_str=None):
        """오늘의 토픽 목록 가져오기"""
        params = {}
        if date_str:
            params["date"] = date_str

        try:
            response = requests.get(f"{self.base_url}/api/topics", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("data", data.get("items", data.get("topics", [])))
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] 토픽 가져오기 실패: {e}")
            return []

    def fetch_topic_articles(self, topic_id):
        """특정 토픽의 기사 목록 가져오기"""
        try:
            response = requests.get(f"{self.base_url}/api/topics/{topic_id}/articles", timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("data", data.get("items", data.get("articles", [])))
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] 토픽 {topic_id} 기사 가져오기 실패: {e}")
            return []


def analyze_articles(predictor, articles, batch_size=16):
    """
    기사 목록 스탠스 분석

    Args:
        predictor: StancePredictor 인스턴스
        articles: 기사 목록 (content 또는 title 포함)
        batch_size: 배치 크기

    Returns:
        분석 결과가 추가된 기사 목록
    """
    if not articles:
        print("[INFO] 분석할 기사가 없습니다.")
        return []

    # 텍스트 추출 (content가 있으면 content, 없으면 title 사용)
    texts = []
    for article in articles:
        text = article.get("content") or article.get("title") or ""
        texts.append(text)

    # 빈 텍스트 확인
    valid_indices = [i for i, t in enumerate(texts) if t.strip()]
    if not valid_indices:
        print("[WARN] 분석할 텍스트가 없습니다.")
        return articles

    valid_texts = [texts[i] for i in valid_indices]

    # 배치 예측
    results = predictor.predict_batch(valid_texts, batch_size=batch_size)

    # 결과 병합
    result_idx = 0
    analyzed_articles = []
    for i, article in enumerate(articles):
        article_copy = article.copy()
        if i in valid_indices:
            analysis = results[result_idx]
            article_copy["stance_analysis"] = analysis
            result_idx += 1
        else:
            article_copy["stance_analysis"] = None
        analyzed_articles.append(article_copy)

    return analyzed_articles


def save_results(articles, output_path, format="json"):
    """분석 결과 저장"""
    if format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
    elif format == "csv":
        # 평탄화
        rows = []
        for article in articles:
            row = {
                "id": article.get("id"),
                "title": article.get("title"),
                "press": article.get("press", {}).get("name") if isinstance(article.get("press"), dict) else article.get("press"),
                "published_at": article.get("published_at"),
            }
            if article.get("stance_analysis"):
                row.update(article["stance_analysis"])
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[OK] 결과 저장: {output_path}")


def print_analysis_summary(articles):
    """분석 결과 요약 출력"""
    total = len(articles)
    analyzed = sum(1 for a in articles if a.get("stance_analysis"))

    print("\n" + "=" * 60)
    print("스탠스 분석 결과 요약")
    print("=" * 60)
    print(f"전체 기사: {total}개")
    print(f"분석 완료: {analyzed}개")

    if analyzed > 0:
        stance_counts = {"support": 0, "neutral": 0, "oppose": 0}
        for article in articles:
            if article.get("stance_analysis"):
                stance = article["stance_analysis"]["predicted_stance"]
                stance_counts[stance] += 1

        print("\n스탠스 분포:")
        labels = {"support": "옹호", "neutral": "중립", "oppose": "비판"}
        for stance, count in stance_counts.items():
            pct = count / analyzed * 100
            bar = "#" * int(pct / 2)
            print(f"  {labels[stance]:4s}: {bar:50s} {count:3d}개 ({pct:.1f}%)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="배포 서버 기사 스탠스 분석")

    # 모델 설정
    parser.add_argument(
        "--model_path",
        type=str,
        default="saved_models/stance_classifier_20251117_125133.pth",
        help="학습된 모델 파일 경로",
    )
    parser.add_argument("--model_name", type=str, default="skt/kobert-base-v1", help="KoBERT 모델명")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")

    # 필터 옵션
    parser.add_argument("--topic_id", type=int, help="특정 토픽 ID")
    parser.add_argument("--date", type=str, help="날짜 필터 (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=50, help="가져올 기사 수")

    # 출력 옵션
    parser.add_argument("--output", type=str, help="결과 저장 경로")
    parser.add_argument("--format", type=str, choices=["json", "csv"], default="json", help="출력 형식")

    # 서버 저장 옵션 (추후 구현)
    parser.add_argument("--save_to_server", action="store_true", help="결과를 서버에 저장 (API 준비 후)")

    args = parser.parse_args()

    # 모델 로드
    print("\n" + "=" * 60)
    print("배포 서버 기사 스탠스 분석")
    print("=" * 60)

    predictor = StancePredictor(model_path=args.model_path, model_name=args.model_name)

    # 기사 가져오기
    fetcher = ArticleFetcher()

    print("\n기사 가져오는 중...")
    if args.topic_id:
        articles = fetcher.fetch_topic_articles(args.topic_id)
        print(f"토픽 {args.topic_id}에서 {len(articles)}개 기사 가져옴")
    else:
        articles = fetcher.fetch_articles(date_str=args.date, limit=args.limit)
        print(f"{len(articles)}개 기사 가져옴")

    if not articles:
        print("[WARN] 가져온 기사가 없습니다.")
        return

    # 기사 상세 정보 필요 시 가져오기 (content가 없는 경우)
    articles_with_content = []
    need_detail = any(not a.get("content") for a in articles)

    if need_detail:
        print("\n기사 본문 가져오는 중...")
        for article in tqdm(articles, desc="본문 로딩"):
            if not article.get("content"):
                detail = fetcher.fetch_article_detail(article.get("id"))
                if detail:
                    article = {**article, **detail}
            articles_with_content.append(article)
        articles = articles_with_content

    # 스탠스 분석
    print("\n스탠스 분석 시작...")
    analyzed_articles = analyze_articles(predictor, articles, batch_size=args.batch_size)

    # 결과 요약 출력
    print_analysis_summary(analyzed_articles)

    # 결과 저장
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "data/analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/stance_analysis_{timestamp}.{args.format}"

    save_results(analyzed_articles, output_path, format=args.format)

    # 서버 저장 (추후 구현)
    if args.save_to_server:
        print("\n[INFO] 서버 저장 기능은 백엔드 API 준비 후 구현 예정입니다.")
        # TODO: 백엔드 팀과 협의 후 구현
        # for article in analyzed_articles:
        #     if article.get("stance_analysis"):
        #         response = requests.post(
        #             f"{API_BASE_URL}/api/articles/{article['id']}/stance",
        #             json=article["stance_analysis"]
        #         )


if __name__ == "__main__":
    main()

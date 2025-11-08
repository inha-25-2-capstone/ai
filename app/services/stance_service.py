"""
스탠스 분석 서비스
뉴스 기사의 스탠스를 분석하는 비즈니스 로직
"""

import logging
import os

from app.models import StancePredictor

logger = logging.getLogger(__name__)


class StanceService:
    """스탠스 분석 서비스"""

    def __init__(self, model_path=None):
        """
        초기화

        Args:
            model_path: 학습된 모델 경로 (None이면 새 모델 사용)
        """
        # 모델 파일이 없으면 초기화하지 않음 (학습 전)
        if model_path and os.path.exists(model_path):
            self.predictor = StancePredictor(model_path=model_path)
            logger.info(f"StanceService initialized with device: {self.predictor.device}")
        else:
            self.predictor = None
            if model_path:
                logger.warning(f"Model file not found at {model_path}. Service will run in no-model mode.")
            else:
                logger.warning("No model path provided. Service will run in no-model mode.")

    def analyze_article(self, article_text):
        """
        단일 기사의 스탠스 분석

        Args:
            article_text: 기사 텍스트

        Returns:
            dict: 스탠스 분석 결과
        """
        try:
            if not article_text or not article_text.strip():
                raise ValueError("기사 텍스트가 비어있습니다.")

            # 모델이 없으면 더미 응답 반환
            if self.predictor is None:
                logger.warning("Model not loaded. Returning dummy response.")
                return {
                    "stance": "중립",
                    "stance_id": 1,
                    "confidence": 0.33,
                    "probabilities": {"옹호": 0.33, "중립": 0.34, "비판": 0.33},
                    "note": "Model not trained yet. This is a dummy response.",
                }

            result = self.predictor.predict(article_text)
            logger.info(f"Article analyzed: stance={result['stance']}, confidence={result['confidence']}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing article: {str(e)}")
            raise

    def analyze_articles_batch(self, articles, batch_size=16):
        """
        여러 기사의 스탠스를 배치로 분석

        Args:
            articles: 기사 텍스트 리스트 또는 기사 딕셔너리 리스트
            batch_size: 배치 크기

        Returns:
            list: 스탠스 분석 결과 리스트
        """
        try:
            # 기사 텍스트 추출
            if isinstance(articles[0], dict):
                texts = [article.get("content", "") or article.get("text", "") for article in articles]
            else:
                texts = articles

            # 빈 텍스트 확인
            if not all(text.strip() for text in texts):
                raise ValueError("비어있는 기사 텍스트가 포함되어 있습니다.")

            # 모델이 없으면 더미 응답 반환
            if self.predictor is None:
                logger.warning("Model not loaded. Returning dummy responses.")
                return [
                    {
                        "stance": "중립",
                        "stance_id": 1,
                        "confidence": 0.33,
                        "probabilities": {"옹호": 0.33, "중립": 0.34, "비판": 0.33},
                        "note": "Model not trained yet. This is a dummy response.",
                    }
                    for _ in texts
                ]

            results = self.predictor.predict_batch(texts, batch_size=batch_size)
            logger.info(f"Batch analysis completed: {len(results)} articles")

            return results

        except Exception as e:
            logger.error(f"Error in batch analysis: {str(e)}")
            raise

    def analyze_topic_articles(self, topic_articles):
        """
        토픽별 기사들의 스탠스 분석 및 그룹화

        Args:
            topic_articles: {
                'topic_id': topic_id,
                'articles': [기사 딕셔너리 리스트]
            }

        Returns:
            dict: {
                'topic_id': topic_id,
                'articles_by_stance': {
                    '옹호': [기사 리스트],
                    '중립': [기사 리스트],
                    '비판': [기사 리스트]
                },
                'stance_distribution': {
                    '옹호': count,
                    '중립': count,
                    '비판': count
                }
            }
        """
        try:
            topic_id = topic_articles["topic_id"]
            articles = topic_articles["articles"]

            # 스탠스 분석
            stance_results = self.analyze_articles_batch(articles)

            # 스탠스별로 기사 그룹화
            articles_by_stance = {"옹호": [], "중립": [], "비판": []}

            stance_distribution = {"옹호": 0, "중립": 0, "비판": 0}

            for i, (article, stance_result) in enumerate(zip(articles, stance_results)):
                stance = stance_result["stance"]
                article_with_stance = {
                    **article,
                    "stance": stance,
                    "stance_id": stance_result["stance_id"],
                    "confidence": stance_result["confidence"],
                }

                articles_by_stance[stance].append(article_with_stance)
                stance_distribution[stance] += 1

            logger.info(f"Topic {topic_id} analyzed: {stance_distribution}")

            return {
                "topic_id": topic_id,
                "articles_by_stance": articles_by_stance,
                "stance_distribution": stance_distribution,
            }

        except Exception as e:
            logger.error(f"Error analyzing topic articles: {str(e)}")
            raise

    def get_diverse_articles(self, topic_analysis, representative_article_id):
        """
        대표 기사와 다른 관점의 기사 추출

        Args:
            topic_analysis: analyze_topic_articles의 결과
            representative_article_id: 대표 기사 ID

        Returns:
            dict: {
                'representative_article': 대표 기사,
                'support_articles': 옹호 기사 리스트,
                'neutral_articles': 중립 기사 리스트,
                'oppose_articles': 비판 기사 리스트
            }
        """
        try:
            articles_by_stance = topic_analysis["articles_by_stance"]

            # 대표 기사 찾기
            representative_article = None
            representative_stance = None

            for stance, articles in articles_by_stance.items():
                for article in articles:
                    if (
                        article.get("id") == representative_article_id
                        or article.get("article_id") == representative_article_id
                    ):
                        representative_article = article
                        representative_stance = stance
                        break
                if representative_article:
                    break

            if not representative_article:
                raise ValueError(f"Representative article {representative_article_id} not found")

            # 다양한 관점의 기사 선정
            result = {
                "representative_article": representative_article,
                "support_articles": articles_by_stance["옹호"],
                "neutral_articles": articles_by_stance["중립"],
                "oppose_articles": articles_by_stance["비판"],
            }

            # 대표 기사와 같은 스탠스의 경우 대표 기사 제외
            if representative_stance:
                stance_key = f"{representative_stance.lower()}_articles"
                if stance_key in result:
                    result[stance_key] = [
                        a
                        for a in result[stance_key]
                        if a.get("id") != representative_article_id and a.get("article_id") != representative_article_id
                    ]

            return result

        except Exception as e:
            logger.error(f"Error getting diverse articles: {str(e)}")
            raise

"""
API 스키마
Pydantic 모델 정의
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class StanceResult(BaseModel):
    """스탠스 분석 결과"""
    stance: str = Field(..., description="스탠스: 옹호/중립/비판")
    stance_id: int = Field(..., description="스탠스 ID (0: 옹호, 1: 중립, 2: 비판)")
    confidence: float = Field(..., description="신뢰도")
    probabilities: Dict[str, float] = Field(..., description="각 클래스별 확률")

    class Config:
        json_schema_extra = {
            "example": {
                "stance": "옹호",
                "stance_id": 0,
                "confidence": 0.95,
                "probabilities": {
                    "옹호": 0.95,
                    "중립": 0.03,
                    "비판": 0.02
                }
            }
        }


class AnalyzeRequest(BaseModel):
    """단일 기사 분석 요청"""
    text: str = Field(..., description="기사 본문", min_length=10)

    class Config:
        json_schema_extra = {
            "example": {
                "text": "정부의 새로운 정책이 발표되었습니다..."
            }
        }


class ArticleInput(BaseModel):
    """기사 입력 데이터"""
    article_id: Optional[int] = Field(None, description="기사 ID")
    text: str = Field(..., description="기사 본문", min_length=10)
    title: Optional[str] = Field(None, description="기사 제목")

    class Config:
        json_schema_extra = {
            "example": {
                "article_id": 1,
                "text": "정부의 새로운 정책이 발표되었습니다...",
                "title": "정부, 새로운 정책 발표"
            }
        }


class BatchAnalyzeRequest(BaseModel):
    """배치 분석 요청"""
    articles: List[ArticleInput] = Field(..., description="기사 리스트")
    batch_size: int = Field(16, description="배치 크기", ge=1, le=32)

    class Config:
        json_schema_extra = {
            "example": {
                "articles": [
                    {"article_id": 1, "text": "기사1 본문..."},
                    {"article_id": 2, "text": "기사2 본문..."}
                ],
                "batch_size": 16
            }
        }


class BatchAnalyzeResponse(BaseModel):
    """배치 분석 응답"""
    results: List[StanceResult]

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "stance": "옹호",
                        "stance_id": 0,
                        "confidence": 0.95,
                        "probabilities": {"옹호": 0.95, "중립": 0.03, "비판": 0.02}
                    }
                ]
            }
        }


class TopicAnalyzeRequest(BaseModel):
    """토픽 분석 요청"""
    topic_id: int = Field(..., description="토픽 ID")
    articles: List[ArticleInput] = Field(..., description="토픽에 속한 기사 리스트")

    class Config:
        json_schema_extra = {
            "example": {
                "topic_id": 1,
                "articles": [
                    {"article_id": 1, "text": "기사1 본문..."},
                    {"article_id": 2, "text": "기사2 본문..."}
                ]
            }
        }


class ArticleWithStance(BaseModel):
    """스탠스가 포함된 기사"""
    article_id: Optional[int]
    text: str
    title: Optional[str]
    stance: str
    stance_id: int
    confidence: float


class TopicAnalyzeResponse(BaseModel):
    """토픽 분석 응답"""
    topic_id: int
    articles_by_stance: Dict[str, List[ArticleWithStance]]
    stance_distribution: Dict[str, int]

    class Config:
        json_schema_extra = {
            "example": {
                "topic_id": 1,
                "articles_by_stance": {
                    "옹호": [],
                    "중립": [],
                    "비판": []
                },
                "stance_distribution": {
                    "옹호": 5,
                    "중립": 3,
                    "비판": 2
                }
            }
        }


class SaveStanceRequest(BaseModel):
    """스탠스 분석 결과 저장 요청"""
    article_id: int = Field(..., description="기사 ID")
    support_prob: float = Field(..., description="옹호 확률", ge=0, le=1)
    neutral_prob: float = Field(..., description="중립 확률", ge=0, le=1)
    oppose_prob: float = Field(..., description="비판 확률", ge=0, le=1)
    final_stance: str = Field(..., description="최종 스탠스: support, neutral, oppose")
    confidence_score: float = Field(..., description="신뢰도", ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "article_id": 1,
                "support_prob": 0.95,
                "neutral_prob": 0.03,
                "oppose_prob": 0.02,
                "final_stance": "support",
                "confidence_score": 0.95
            }
        }


class BatchSaveStanceRequest(BaseModel):
    """배치 스탠스 저장 요청"""
    analyses: List[SaveStanceRequest] = Field(..., description="스탠스 분석 결과 리스트")

    class Config:
        json_schema_extra = {
            "example": {
                "analyses": [
                    {
                        "article_id": 1,
                        "support_prob": 0.95,
                        "neutral_prob": 0.03,
                        "oppose_prob": 0.02,
                        "final_stance": "support",
                        "confidence_score": 0.95
                    }
                ]
            }
        }


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    service: str
    timestamp: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "AI News Stance Analysis",
                "timestamp": "2024-01-01T00:00:00"
            }
        }


class ErrorResponse(BaseModel):
    """에러 응답"""
    error: str
    detail: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Internal server error",
                "detail": "스탠스 분석 중 오류가 발생했습니다."
            }
        }

"""
API 라우터
스탠스 분석 관련 API 엔드포인트
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.schemas import (
    AnalyzeRequest,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    BatchSaveStanceRequest,
    ErrorResponse,
    HealthResponse,
    SaveStanceRequest,
    StanceResult,
    TopicAnalyzeRequest,
    TopicAnalyzeResponse,
)
from app.database import Article, StanceAnalysis, get_db
from app.services import StanceService

logger = logging.getLogger(__name__)

# API 라우터 생성
router = APIRouter(prefix="/api", tags=["stance-analysis"])

# StanceService 인스턴스 (전역)
stance_service = None


def get_stance_service():
    """StanceService 의존성"""
    if stance_service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Stance service not initialized")
    return stance_service


def init_stance_service(model_path=None, repo_id=None):
    """StanceService 초기화"""
    global stance_service
    stance_service = StanceService(model_path=model_path, repo_id=repo_id)
    logger.info("StanceService initialized in routes")


@router.get("/health", response_model=HealthResponse, summary="헬스 체크", description="API 서버 상태 확인")
async def health_check():
    """헬스 체크 엔드포인트"""
    return HealthResponse(status="healthy", service="AI News Stance Analysis", timestamp=datetime.utcnow())


@router.post(
    "/analyze",
    response_model=StanceResult,
    summary="단일 기사 스탠스 분석",
    description="단일 뉴스 기사의 스탠스를 분석합니다 (옹호/중립/비판)",
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        500: {"model": ErrorResponse, "description": "서버 오류"},
    },
)
async def analyze_article(request: AnalyzeRequest, service: StanceService = Depends(get_stance_service)):
    """
    단일 기사 스탠스 분석
    """
    try:
        result = service.analyze_article(request.text)
        return StanceResult(**result)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error in analyze_article: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="스탠스 분석 중 오류가 발생했습니다."
        )


@router.post(
    "/analyze/batch",
    response_model=BatchAnalyzeResponse,
    summary="배치 스탠스 분석",
    description="여러 뉴스 기사의 스탠스를 배치로 분석합니다",
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        500: {"model": ErrorResponse, "description": "서버 오류"},
    },
)
async def analyze_articles_batch(request: BatchAnalyzeRequest, service: StanceService = Depends(get_stance_service)):
    """
    여러 기사 배치 스탠스 분석
    """
    try:
        # 기사 텍스트 추출
        articles = [{"text": article.text} for article in request.articles]

        results = service.analyze_articles_batch(articles, batch_size=request.batch_size)

        return BatchAnalyzeResponse(results=[StanceResult(**result) for result in results])

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error in analyze_articles_batch: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="배치 스탠스 분석 중 오류가 발생했습니다."
        )


@router.post(
    "/analyze/topic",
    response_model=TopicAnalyzeResponse,
    summary="토픽별 기사 스탠스 분석",
    description="토픽에 속한 모든 기사의 스탠스를 분석하고 그룹화합니다",
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        500: {"model": ErrorResponse, "description": "서버 오류"},
    },
)
async def analyze_topic(request: TopicAnalyzeRequest, service: StanceService = Depends(get_stance_service)):
    """
    토픽별 기사 스탠스 분석 및 그룹화
    """
    try:
        topic_data = {
            "topic_id": request.topic_id,
            "articles": [
                {"article_id": article.article_id, "text": article.text, "title": article.title}
                for article in request.articles
            ],
        }

        result = service.analyze_topic_articles(topic_data)

        return TopicAnalyzeResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error in analyze_topic: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="토픽 스탠스 분석 중 오류가 발생했습니다."
        )


@router.post(
    "/stance/save",
    status_code=status.HTTP_201_CREATED,
    summary="스탠스 분석 결과 저장",
    description="스탠스 분석 결과를 데이터베이스에 저장합니다",
    responses={
        201: {"description": "저장 성공"},
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        404: {"model": ErrorResponse, "description": "기사를 찾을 수 없음"},
        500: {"model": ErrorResponse, "description": "서버 오류"},
    },
)
async def save_stance_analysis(request: SaveStanceRequest, db: Session = Depends(get_db)):
    """
    스탠스 분석 결과를 데이터베이스에 저장
    """
    try:
        # 기사 존재 확인
        article = db.query(Article).filter(Article.article_id == request.article_id).first()
        if not article:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Article {request.article_id} not found")

        # 기존 분석 결과 확인
        existing_analysis = db.query(StanceAnalysis).filter(StanceAnalysis.article_id == request.article_id).first()

        if existing_analysis:
            # 업데이트
            existing_analysis.support_prob = request.support_prob
            existing_analysis.neutral_prob = request.neutral_prob
            existing_analysis.oppose_prob = request.oppose_prob
            existing_analysis.final_stance = request.final_stance
            existing_analysis.confidence_score = request.confidence_score
            existing_analysis.created_at = datetime.utcnow()
        else:
            # 새로 생성
            new_analysis = StanceAnalysis(
                article_id=request.article_id,
                support_prob=request.support_prob,
                neutral_prob=request.neutral_prob,
                oppose_prob=request.oppose_prob,
                final_stance=request.final_stance,
                confidence_score=request.confidence_score,
            )
            db.add(new_analysis)

        db.commit()

        return {"message": "Stance analysis saved successfully"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving stance analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="스탠스 분석 결과 저장 중 오류가 발생했습니다."
        )


@router.post(
    "/stance/save/batch",
    status_code=status.HTTP_201_CREATED,
    summary="배치 스탠스 분석 결과 저장",
    description="여러 스탠스 분석 결과를 데이터베이스에 배치로 저장합니다",
    responses={
        201: {"description": "저장 성공"},
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        500: {"model": ErrorResponse, "description": "서버 오류"},
    },
)
async def save_stance_analysis_batch(request: BatchSaveStanceRequest, db: Session = Depends(get_db)):
    """
    배치 스탠스 분석 결과를 데이터베이스에 저장
    """
    try:
        saved_count = 0
        updated_count = 0

        for analysis in request.analyses:
            # 기사 존재 확인
            article = db.query(Article).filter(Article.article_id == analysis.article_id).first()
            if not article:
                logger.warning(f"Article {analysis.article_id} not found, skipping...")
                continue

            # 기존 분석 결과 확인
            existing_analysis = (
                db.query(StanceAnalysis).filter(StanceAnalysis.article_id == analysis.article_id).first()
            )

            if existing_analysis:
                # 업데이트
                existing_analysis.support_prob = analysis.support_prob
                existing_analysis.neutral_prob = analysis.neutral_prob
                existing_analysis.oppose_prob = analysis.oppose_prob
                existing_analysis.final_stance = analysis.final_stance
                existing_analysis.confidence_score = analysis.confidence_score
                existing_analysis.created_at = datetime.utcnow()
                updated_count += 1
            else:
                # 새로 생성
                new_analysis = StanceAnalysis(
                    article_id=analysis.article_id,
                    support_prob=analysis.support_prob,
                    neutral_prob=analysis.neutral_prob,
                    oppose_prob=analysis.oppose_prob,
                    final_stance=analysis.final_stance,
                    confidence_score=analysis.confidence_score,
                )
                db.add(new_analysis)
                saved_count += 1

        db.commit()

        return {
            "message": "Batch stance analysis saved successfully",
            "saved": saved_count,
            "updated": updated_count,
            "total": saved_count + updated_count,
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error in batch save: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="배치 스탠스 분석 결과 저장 중 오류가 발생했습니다.",
        )


@router.get(
    "/stance/{article_id}",
    summary="스탠스 분석 결과 조회",
    description="특정 기사의 스탠스 분석 결과를 조회합니다",
    responses={
        404: {"model": ErrorResponse, "description": "분석 결과를 찾을 수 없음"},
        500: {"model": ErrorResponse, "description": "서버 오류"},
    },
)
async def get_stance_analysis(article_id: int, db: Session = Depends(get_db)):
    """
    특정 기사의 스탠스 분석 결과 조회
    """
    try:
        analysis = db.query(StanceAnalysis).filter(StanceAnalysis.article_id == article_id).first()

        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Stance analysis for article {article_id} not found"
            )

        return {
            "article_id": analysis.article_id,
            "support_prob": analysis.support_prob,
            "neutral_prob": analysis.neutral_prob,
            "oppose_prob": analysis.oppose_prob,
            "final_stance": analysis.final_stance,
            "confidence_score": analysis.confidence_score,
            "created_at": analysis.created_at,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stance analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="스탠스 분석 결과 조회 중 오류가 발생했습니다."
        )

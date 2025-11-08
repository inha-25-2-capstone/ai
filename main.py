"""
AI 기반 객관적 뉴스 추천 서비스 - AI 파트
FastAPI 애플리케이션 메인 파일
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import init_stance_service, router

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 시작/종료 시 실행되는 이벤트
    """
    # 시작 시
    logger.info("Starting AI News Stance Analysis Service...")

    # 데이터베이스 테이블 생성 (개발 환경에서만, 프로덕션에서는 마이그레이션 사용)
    # if engine is not None:
    #     Base.metadata.create_all(bind=engine)
    #     logger.info("Database tables created")

    # 스탠스 분석 모델 초기화
    model_path = os.getenv("MODEL_PATH", None)
    init_stance_service(model_path=model_path)
    logger.info("Stance service initialized")

    yield

    # 종료 시
    logger.info("Shutting down AI News Stance Analysis Service...")


# FastAPI 애플리케이션 생성
app = FastAPI(
    title="AI News Stance Analysis API",
    description="AI 기반 뉴스 스탠스 분석 서비스 (옹호/중립/비판 분류)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(router)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "AI News Stance Analysis",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "analyze": "/api/analyze",
            "batch_analyze": "/api/analyze/batch",
            "topic_analyze": "/api/analyze/topic",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    # 개발 서버 실행
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info")  # 개발 모드에서만 True

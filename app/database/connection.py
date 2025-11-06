"""
데이터베이스 연결 설정
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 데이터베이스 URL 가져오기
DATABASE_URL = os.getenv('DATABASE_URL')

engine = None
SessionLocal = None

if not DATABASE_URL:
    # 개별 환경 변수로 URL 구성
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME')

    if all([DB_USER, DB_PASSWORD, DB_NAME]):
        DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# DB 연결 설정 (옵셔널)
if DATABASE_URL:
    try:
        # SQLAlchemy 엔진 생성
        engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False  # 개발 시 True로 설정하면 SQL 쿼리 로그 출력
        )

        # 세션 팩토리 생성
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("Database connection configured successfully")
    except Exception as e:
        logger.warning(f"Failed to configure database: {e}")
        engine = None
        SessionLocal = None
else:
    logger.warning("Database connection not configured. DB-related endpoints will not work.")


def get_db():
    """
    데이터베이스 세션 의존성
    FastAPI에서 사용
    """
    if SessionLocal is None:
        raise RuntimeError("Database not configured. Please set DATABASE_URL in .env file.")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

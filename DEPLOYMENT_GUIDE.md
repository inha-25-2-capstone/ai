# Hugging Face Spaces 배포 가이드

AI 기반 뉴스 스탠스 분석 API를 Hugging Face Spaces에 배포하는 방법을 설명합니다.

## 목차
1. [사전 준비](#사전-준비)
2. [배포 파일 준비](#배포-파일-준비)
3. [Hugging Face Space 생성](#hugging-face-space-생성)
4. [환경 변수 설정](#환경-변수-설정)
5. [배포 및 테스트](#배포-및-테스트)
6. [문제 해결](#문제-해결)

---

## 사전 준비

### 1. Hugging Face 계정 생성
- [Hugging Face](https://huggingface.co/) 가입
- 로그인 후 프로필 설정 확인

### 2. Git 설치 확인
```bash
git --version
```

### 3. Hugging Face CLI 설치 (선택사항)
```bash
pip install huggingface-hub
huggingface-cli login
```

---

## 배포 파일 준비

### 1. Dockerfile 작성

프로젝트 루트에 `Dockerfile` 생성:

```dockerfile
# Python 3.10 이미지 사용 (torch 호환성)
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /home/user/app

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 사용자 생성 (Hugging Face Spaces는 UID 1000으로 실행)
RUN useradd -m -u 1000 user

# Python 의존성 파일 복사 및 설치
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY --chown=user . .

# 사용자 전환
USER user

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV HOST=0.0.0.0

# 포트 노출
EXPOSE 7860

# FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 2. README.md 수정

프로젝트 루트의 `README.md` 파일 상단에 YAML 헤더 추가:

```markdown
---
title: AI News Stance Analysis
emoji: 📰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# AI News Stance Analysis API

뉴스 기사의 스탠스(옹호/중립/비판)를 분석하는 FastAPI 기반 AI 서비스입니다.

## 주요 기능
- 단일 기사 스탠스 분석
- 배치 분석
- 토픽별 기사 그룹 분석

## API 엔드포인트
- `GET /` - 서비스 정보
- `GET /api/health` - 헬스 체크
- `POST /api/analyze` - 단일 기사 분석
- `POST /api/analyze/batch` - 배치 분석
- `POST /api/analyze/topic` - 토픽별 분석
- `GET /docs` - API 문서 (Swagger UI)

## 사용 예시

### 단일 기사 분석
```bash
curl -X POST "https://your-space-name.hf.space/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "분석할 기사 본문"}'
```

### 응답 예시
```json
{
  "stance": "neutral",
  "support_prob": 0.25,
  "neutral_prob": 0.60,
  "oppose_prob": 0.15,
  "confidence": 0.60
}
```

## 기술 스택
- **Framework**: FastAPI
- **ML**: PyTorch, Transformers, KoBERT
- **Database**: PostgreSQL (Optional)

## 로컬 실행
```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env

# 서버 실행
python main.py
```

## License
MIT License
```

### 3. requirements.txt 최적화

배포용으로 `requirements.txt` 검토 및 최적화:

```txt
# FastAPI & Web
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# AI/ML (CPU 버전 - Hugging Face Spaces는 기본적으로 CPU)
torch==2.1.0
transformers==4.35.0
kobert-transformers==0.5.1
sentencepiece==0.1.99

# Data Processing
pandas==2.0.3
numpy==1.24.3

# Utils
python-dotenv==1.0.0
requests==2.31.0

# Database (옵션 - 필요시 주석 해제)
# psycopg2-binary==2.9.9
# SQLAlchemy==2.0.23
```

**참고**: Hugging Face Spaces 무료 티어는 CPU 인스턴스를 제공합니다. GPU가 필요한 경우 유료 티어를 사용해야 합니다.

### 4. .dockerignore 생성

불필요한 파일이 이미지에 포함되지 않도록 설정:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo

# Git
.git/
.gitignore

# Environment
.env
.env.local

# Data
data/raw/
data/processed/
*.csv
*.json

# Models (학습된 모델이 큰 경우)
# saved_models/

# Notebooks
notebooks/
*.ipynb

# Tests
tests/
pytest_cache/

# Documentation
docs/
*.md
!README.md

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db
```

### 5. 모델 파일 준비

**옵션 A: 모델 파일 포함 (작은 모델)**
- `saved_models/` 디렉토리를 그대로 포함
- `.dockerignore`에서 `saved_models/` 주석 처리

**옵션 B: 모델 다운로드 (큰 모델 - 권장)**
- Hugging Face Hub에 모델 업로드
- 런타임에 다운로드하도록 코드 수정

`app/models/stance_classifier.py` 수정 예시:
```python
from transformers import AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

def load_model(model_name_or_path):
    """모델 로드 (Hugging Face Hub에서 다운로드)"""
    if model_name_or_path.startswith("hf://"):
        # Hugging Face Hub에서 다운로드
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path.replace("hf://", "")
        )
    else:
        # 로컬 파일 로드
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        )
    return model
```

환경 변수 설정:
```bash
MODEL_PATH=hf://your-username/your-model-name
```

---

## Hugging Face Space 생성

### 1. Space 생성

1. [Hugging Face Spaces](https://huggingface.co/spaces)로 이동
2. **"Create new Space"** 클릭
3. 정보 입력:
   - **Space name**: `ai-news-stance-analysis` (원하는 이름)
   - **License**: MIT
   - **SDK**: **Docker** 선택 ⭐
   - **Visibility**: Public or Private
4. **"Create Space"** 클릭

### 2. Git 저장소 복제

Space가 생성되면 Git 저장소 URL이 제공됩니다:

```bash
# Space 저장소 복제
git clone https://huggingface.co/spaces/your-username/ai-news-stance-analysis
cd ai-news-stance-analysis
```

### 3. 파일 복사

프로젝트 파일을 Space 저장소로 복사:

```bash
# 프로젝트 루트에서
cp -r app/ ../ai-news-stance-analysis/
cp main.py ../ai-news-stance-analysis/
cp requirements.txt ../ai-news-stance-analysis/
cp Dockerfile ../ai-news-stance-analysis/
cp .dockerignore ../ai-news-stance-analysis/
cp README.md ../ai-news-stance-analysis/

# 모델 파일 복사 (옵션 A 선택 시)
cp -r saved_models/ ../ai-news-stance-analysis/
```

또는 전체 프로젝트를 직접 Space로 사용:

```bash
# 기존 프로젝트에 Hugging Face Space remote 추가
cd /path/to/your/ai-project
git remote add space https://huggingface.co/spaces/your-username/ai-news-stance-analysis

# 브랜치 푸시
git push space main
```

### 4. 커밋 및 푸시

```bash
cd ai-news-stance-analysis

# 파일 추가
git add .

# 커밋
git commit -m "Initial deployment to Hugging Face Spaces"

# 푸시
git push
```

푸시하면 자동으로 빌드가 시작됩니다.

---

## 환경 변수 설정

### Hugging Face Secrets 사용

Space 설정에서 환경 변수를 안전하게 관리할 수 있습니다.

1. **Space 페이지**로 이동
2. **Settings** 탭 클릭
3. **Variables and secrets** 섹션에서 환경 변수 추가:

| 변수 이름 | 값 | 설명 |
|-----------|-----|------|
| `MODEL_PATH` | `hf://username/model-name` 또는 `./saved_models/best_model` | 모델 경로 |
| `DATABASE_URL` | `postgresql://...` | DB 연결 문자열 (옵션) |
| `CORS_ORIGINS` | `*` | CORS 허용 도메인 |
| `LOG_LEVEL` | `INFO` | 로그 레벨 |

**주의**: DB 비밀번호 등 민감한 정보는 반드시 Secrets로 설정하세요.

### main.py 환경 변수 처리 확인

```python
import os
from dotenv import load_dotenv

load_dotenv()  # 로컬에서만 동작, Spaces에서는 Secrets 사용

MODEL_PATH = os.getenv("MODEL_PATH", "./saved_models/best_model")
DATABASE_URL = os.getenv("DATABASE_URL", None)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
```

---

## 배포 및 테스트

### 1. 빌드 상태 확인

Space 페이지의 **"Logs"** 탭에서 빌드 진행 상황 확인:
- Docker 이미지 빌드
- 의존성 설치
- 애플리케이션 시작

### 2. 빌드 완료 후 테스트

Space URL이 활성화되면 (예: `https://your-username-ai-news-stance-analysis.hf.space`):

#### a) 브라우저 테스트
```
https://your-username-ai-news-stance-analysis.hf.space
```

루트 엔드포인트 응답 확인:
```json
{
  "service": "AI News Stance Analysis",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {...}
}
```

#### b) API 문서 확인
```
https://your-username-ai-news-stance-analysis.hf.space/docs
```

Swagger UI에서 API 테스트 가능

#### c) 헬스 체크
```bash
curl https://your-username-ai-news-stance-analysis.hf.space/api/health
```

#### d) 스탠스 분석 테스트
```bash
curl -X POST "https://your-username-ai-news-stance-analysis.hf.space/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "정부의 새로운 정책은 경제 성장에 큰 도움이 될 것으로 전망된다."
  }'
```

### 3. 성능 모니터링

Space 페이지의 **"Community"** 탭에서:
- 실시간 로그 확인
- 메모리 사용량
- CPU 사용량
- 응답 시간

---

## 문제 해결

### 빌드 실패

#### 1. 의존성 설치 실패
**증상**: `pip install` 중 에러 발생

**해결**:
```dockerfile
# Dockerfile에 시스템 패키지 추가
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
```

#### 2. 권한 오류
**증상**: `Permission denied` 에러

**해결**:
```dockerfile
# COPY 시 --chown=user 사용
COPY --chown=user . /home/user/app

# 디렉토리 권한 설정
RUN chown -R user:user /home/user/app
```

#### 3. 포트 충돌
**증상**: 서버가 시작되지 않음

**해결**:
```dockerfile
# README.md YAML 헤더 확인
---
sdk: docker
app_port: 7860
---

# Dockerfile CMD 확인
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 런타임 에러

#### 1. 모델 로드 실패
**증상**: `Model not found` 에러

**해결**:
- Hugging Face Secrets에 `MODEL_PATH` 설정
- 모델 파일이 이미지에 포함되었는지 확인
- `.dockerignore`에서 `saved_models/` 주석 해제

#### 2. 메모리 부족
**증상**: `OOMKilled` 또는 서버 종료

**해결**:
- 모델 크기 최적화 (quantization, pruning)
- 배치 크기 줄이기
- 유료 GPU 인스턴스 사용 고려

#### 3. CORS 에러
**증상**: 프론트엔드에서 API 호출 실패

**해결**:
```python
# main.py CORS 설정 확인
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 특정 도메인
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 로그 확인

Space의 **"Logs"** 탭에서 실시간 로그 확인:
```bash
# 로컬에서 Hugging Face CLI 사용
huggingface-cli logs your-username/ai-news-stance-analysis
```

---

## 배포 최적화

### 1. Docker 이미지 크기 최적화

```dockerfile
# Multi-stage build 사용
FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.10-slim

WORKDIR /home/user/app
RUN useradd -m -u 1000 user

# 빌더에서 Python 패키지만 복사
COPY --from=builder /root/.local /home/user/.local
COPY --chown=user . .

USER user

ENV PATH=/home/user/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 2. 캐싱 활용

```dockerfile
# requirements.txt를 먼저 복사하여 레이어 캐싱
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드는 나중에 복사 (자주 변경되므로)
COPY --chown=user . .
```

### 3. 불필요한 파일 제외

`.dockerignore` 철저히 관리하여 이미지 크기 줄이기

---

## 추가 리소스

- [Hugging Face Spaces 공식 문서](https://huggingface.co/docs/hub/spaces)
- [Docker SDK 가이드](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Secrets 관리](https://huggingface.co/docs/hub/spaces-overview#managing-secrets)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)

---

## 다음 단계

배포 후 고려할 사항:

1. **모니터링 설정**
   - Logging 강화
   - 에러 트래킹 (Sentry 등)

2. **성능 최적화**
   - 모델 양자화
   - 캐싱 전략
   - 비동기 처리

3. **CI/CD 구축**
   - GitHub Actions로 자동 배포
   - 테스트 자동화

4. **비용 관리**
   - 무료 티어 한계 확인
   - 유료 옵션 검토 (GPU, 더 많은 메모리)

---

**배포 문의 및 이슈**: [GitHub Issues](https://github.com/your-repo/issues)

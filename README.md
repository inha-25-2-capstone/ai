# AI 기반 객관적 뉴스 추천 서비스 - AI 파트

[![CI](https://github.com/inha-25-2-capstone/ai/actions/workflows/ci.yml/badge.svg)](https://github.com/inha-25-2-capstone/ai/actions/workflows/ci.yml)

국내 정치 뉴스의 언론사별 편향성을 극복하고, 같은 이슈에 대해 옹호/중립/비판의 다양한 관점을 제공하는 AI 기반 뉴스 스탠스 분석 시스템

## 주요 기능

### 1. 스탠스 분석
- **KoBERT 기반 기사 분류**: 뉴스 기사를 옹호/중립/비판 3개 클래스로 분류
- **배치 처리**: 여러 기사를 효율적으로 분석
- **토픽별 그룹화**: 같은 토픽 내 기사들을 스탠스별로 분류

### 2. API 서비스
- FastAPI 기반 REST API
- PostgreSQL 데이터베이스 연동
- 스탠스 분석 결과 저장 및 조회

## 기술 스택

- **Backend**: FastAPI + Uvicorn
- **Database**: PostgreSQL + SQLAlchemy
- **AI/ML**: PyTorch, KoBERT (skt/kobert-base-v1)
- **Deployment**: Hugging Face Spaces
- **Development**: Google Colab

## 프로젝트 구조

```
ai/
├── app/
│   ├── api/              # FastAPI 라우터 및 스키마
│   │   ├── __init__.py
│   │   ├── routes.py     # API 엔드포인트
│   │   └── schemas.py    # Pydantic 모델
│   ├── database/         # 데이터베이스 관련
│   │   ├── __init__.py
│   │   ├── models.py     # SQLAlchemy 모델
│   │   └── connection.py # DB 연결 설정
│   ├── models/           # AI 모델
│   │   ├── __init__.py
│   │   └── stance_classifier.py  # KoBERT 스탠스 분류 모델
│   ├── services/         # 비즈니스 로직
│   │   ├── __init__.py
│   │   └── stance_service.py
│   └── utils/            # 유틸리티
├── data/                 # 데이터 파일
├── notebooks/            # Jupyter 노트북
├── saved_models/         # 학습된 모델 저장
├── main.py               # FastAPI 애플리케이션 진입점
├── requirements.txt      # Python 패키지 의존성
├── .env.example          # 환경 변수 예제
└── README.md
```

## 설치 및 실행

### 1. 가상환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

`.env.example` 파일을 `.env`로 복사하고 환경 변수를 설정합니다.

```bash
cp .env.example .env
```

`.env` 파일을 열어 데이터베이스 정보를 입력합니다:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/dbname
```

### 4. 서버 실행

```bash
python main.py
```

또는 uvicorn 직접 실행:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

서버가 실행되면 다음 URL에서 확인할 수 있습니다:
- API 문서: http://localhost:8000/docs
- 대체 문서: http://localhost:8000/redoc

## API 엔드포인트

### 헬스 체크
```
GET /api/health
```

### 단일 기사 스탠스 분석
```
POST /api/analyze
Content-Type: application/json

{
  "text": "기사 본문..."
}
```

**응답**:
```json
{
  "stance": "옹호",
  "stance_id": 0,
  "confidence": 0.95,
  "probabilities": {
    "옹호": 0.95,
    "중립": 0.03,
    "비판": 0.02
  }
}
```

### 배치 스탠스 분석
```
POST /api/analyze/batch
Content-Type: application/json

{
  "articles": [
    {"article_id": 1, "text": "기사1 본문..."},
    {"article_id": 2, "text": "기사2 본문..."}
  ],
  "batch_size": 16
}
```

### 토픽별 스탠스 분석
```
POST /api/analyze/topic
Content-Type: application/json

{
  "topic_id": 1,
  "articles": [...]
}
```

### 스탠스 분석 결과 저장
```
POST /api/stance/save
Content-Type: application/json

{
  "article_id": 1,
  "support_prob": 0.95,
  "neutral_prob": 0.03,
  "oppose_prob": 0.02,
  "final_stance": "support",
  "confidence_score": 0.95
}
```

### 스탠스 분석 결과 조회
```
GET /api/stance/{article_id}
```

## 데이터베이스 스키마

주요 테이블:
- `press`: 언론사 정보
- `article`: 기사 정보
- `topic`: 일일 Top 7 토픽
- `topic_article_mapping`: 토픽-기사 매핑
- `stance_analysis`: 스탠스 분석 결과
- `recommended_article`: 추천 기사

## 모델 학습

데이터 라벨링이 완료되면 다음 단계로 모델을 학습합니다:

1. 라벨링된 데이터 준비
2. 학습 스크립트 실행
3. 학습된 모델을 `saved_models/` 디렉토리에 저장
4. `.env` 파일의 `MODEL_PATH` 업데이트

## 개발 로드맵

- [x] 프로젝트 초기 구조 설정
- [x] KoBERT 스탠스 분류 모델 기본 구조
- [x] FastAPI 서버 구축
- [x] 데이터베이스 연동
- [x] API 엔드포인트 구현
- [ ] 데이터 라벨링 완료
- [ ] 모델 학습 및 평가
- [ ] Hugging Face Spaces 배포
- [ ] 프론트엔드 연동

## 브랜치 전략

### 브랜치 구조
```
main (프로덕션)
  ↑
  └─ develop (개발)
       ↑
       └─ feature/* (기능 개발)
       └─ fix/* (버그 수정)
```

### 개발 워크플로우
1. **새 기능 개발**
   ```bash
   git checkout develop
   git checkout -b feature/기능-이름
   # 개발 후
   git push -u origin feature/기능-이름
   # develop으로 PR 생성
   ```

2. **버그 수정**
   ```bash
   git checkout develop
   git checkout -b fix/버그-이름
   # 수정 후 PR 생성
   ```

3. **배포** (develop → main)
   ```bash
   # develop이 안정화되면 main으로 PR 생성
   ```

### 커밋 메시지 규칙
- `feat`: 새 기능 추가
- `fix`: 버그 수정
- `docs`: 문서 수정
- `style`: 코드 포맷팅
- `refactor`: 리팩토링
- `test`: 테스트 추가/수정
- `ci`: CI 설정 변경

## 주요 용어

- **토픽**: 같은 정치 이슈를 다루는 기사 묶음 (하루 7개)
- **스탠스**: 기사의 논조 (옹호/중립/비판)
- **대표 기사**: 토픽을 대표하는 핵심 기사
- **후보 기사**: 대표 기사와 다른 관점의 기사들

## 라이선스

MIT License

## 팀

인하대학교 2025-2 캡스톤 프로젝트

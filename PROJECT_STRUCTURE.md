# 프로젝트 구조

## 📂 디렉토리 구조

```
ai/
├── scripts/                          # 스크립트 모음
│   ├── batch_api/                    # 배치 API 관련
│   │   ├── openai/                   # OpenAI Batch API
│   │   │   ├── create_batch.py      # 배치 입력 파일 생성
│   │   │   └── analyze_results.py   # 배치 결과 분석
│   │   └── claude/                   # Claude Batch API
│   │       ├── create_batch.py      # 배치 입력 파일 생성
│   │       ├── submit_batch.py      # 배치 제출 및 관리
│   │       └── check_errors.py      # 배치 에러 확인
│   │
│   ├── labeling/                     # 라벨링 관련
│   │   ├── merge_labels.py          # AI 라벨 결합
│   │   ├── auto_labeling.py         # 자동 라벨링
│   │   └── batch_labeling.py        # 배치 라벨링
│   │
│   ├── review/                       # 검수 관련
│   │   ├── extract_disagreements.py # 불일치 데이터 추출
│   │   ├── prepare_samples.py       # 검수 샘플 생성
│   │   └── review_labels.py         # 라벨 검수
│   │
│   ├── data_prep/                    # 데이터 준비
│   │   ├── collect_news.py          # 뉴스 수집
│   │   └── validate_data.py         # 데이터 검증
│   │
│   └── README.md
│
├── data/                             # 데이터 저장소
│   ├── raw/                          # 원본 데이터
│   │   └── bigkinds_summarized.json # 원본 뉴스 데이터
│   │
│   ├── batch_results/                # 배치 처리 결과
│   │   ├── openai/                   # OpenAI 배치 결과
│   │   │   ├── batch_*.jsonl        # 입력/출력 파일
│   │   │   └── batch_info.json      # 배치 정보
│   │   └── claude/                   # Claude 배치 결과
│   │       ├── batch_*.jsonl        # 입력/출력 파일
│   │       └── batch_info_claude.json
│   │
│   ├── labeled/                      # 라벨링된 데이터
│   │   ├── labeled_dataset.json     # 최종 라벨링 데이터
│   │   └── test_samples/            # 테스트 샘플
│   │
│   ├── review/                       # 검수 데이터
│   │   ├── disagreements_for_review.csv   # 검수 대상 (CSV)
│   │   ├── disagreements_for_review.json  # 검수 대상 (JSON)
│   │   └── samples/                       # 검수 샘플
│   │       ├── practice_sample_50.*       # 연습용 샘플
│   │       └── priority1_all.*            # 우선순위 1
│   │
│   └── docs/                         # 가이드 문서
│       ├── AUTO_LABELING_GUIDE.md
│       ├── BATCH_LABELING_GUIDE.md
│       ├── DATA_PREPARATION_GUIDE.md
│       ├── MANUAL_REVIEW_GUIDE.md        # 검수 가이드
│       ├── NEWS_DATA_COLLECTION_GUIDE.md
│       ├── OPENAI_BATCH_GUIDE.md
│       ├── REVIEW_CHECKLIST.md           # 빠른 체크리스트
│       └── TEAM_ASSIGNMENT_TEMPLATE.md   # 팀 배정 템플릿
│
└── ... (기타 프로젝트 파일)
```

## 🎯 주요 워크플로우

### 1. 배치 라벨링

```bash
# OpenAI 배치
python scripts/batch_api/openai/create_batch.py
python scripts/batch_api/openai/analyze_results.py

# Claude 배치
python scripts/batch_api/claude/create_batch.py
python scripts/batch_api/claude/submit_batch.py
```

### 2. 라벨 결합 및 분석

```bash
# OpenAI + Claude 라벨 결합
python scripts/labeling/merge_labels.py

# 불일치 데이터 추출
python scripts/review/extract_disagreements.py
```

### 3. 수동 검수

```bash
# 검수 샘플 생성
python scripts/review/prepare_samples.py

# 검수 파일 위치
data/review/disagreements_for_review.csv
```

## 📖 문서

모든 가이드 문서는 `data/docs/` 디렉토리에 있습니다:

- **MANUAL_REVIEW_GUIDE.md**: 수동 검수 완전 가이드
- **REVIEW_CHECKLIST.md**: 빠른 검수 체크리스트
- **TEAM_ASSIGNMENT_TEMPLATE.md**: 팀 검수 배정 템플릿

## 🔑 설정

API 키는 `.env` 파일에 설정:
```env
ANTHROPIC_API_KEY=your-api-key
OPENAI_API_KEY=your-api-key
```

## 📊 데이터 흐름

```
원본 데이터 (raw/)
    ↓
배치 API 처리 (batch_results/)
    ↓
라벨 결합 (labeled/)
    ↓
불일치 추출 (review/)
    ↓
수동 검수
    ↓
최종 데이터셋
```

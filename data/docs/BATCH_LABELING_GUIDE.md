# Batches API를 이용한 멀티 모델 자동 라벨링 가이드

OpenAI와 Claude의 Batches API를 사용하여 대량의 뉴스 기사를 자동 라벨링하는 방법입니다.

---

## 🎯 개요

### 워크플로우

```
Day 1: 배치 제출
  ├─ OpenAI GPT-4o-mini Batches로 라벨링
  └─ Claude Haiku Batches로 라벨링

Day 2: 결과 확인 (24시간 후)
  ├─ 두 모델 결과 비교
  ├─ 일치: 자동 라벨링 완료 ✅
  └─ 불일치: 검토 필요 ⚠️
```

### 비용 (300개 기사 기준)

| 방법 | 비용 | 할인 |
|------|------|------|
| 일반 API | $0.11 | - |
| **Batches API** | **$0.055** | **50% OFF** ⭐ |

---

## 📋 사전 준비

### 1. API 키 발급

#### OpenAI API 키
1. https://platform.openai.com/api-keys 접속
2. "Create new secret key" 클릭
3. 키 복사

#### Claude API 키
1. https://console.anthropic.com/settings/keys 접속
2. "Create Key" 클릭
3. 키 복사

### 2. 환경 변수 설정

`.env` 파일에 API 키 추가:

```bash
# .env
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. 의존성 설치

```bash
pip install -r requirements-scripts.txt
```

---

## 🚀 사용 방법

### Step 1: 배치 제출

라벨링할 기사 데이터를 준비하고 배치 작업을 시작합니다.

```bash
python scripts/batch_labeling.py \
  --input data/unlabeled_news.csv \
  --output data/batch_labeled.csv \
  --mode submit
```

**입력 파일 형식 (CSV):**
```csv
text,title,source,date,topic
"기사 본문...",기사 제목,조선일보,2024-01-15,부동산정책
"기사 본문...",기사 제목,한겨레,2024-01-15,경제정책
```

**출력:**
```
📝 OpenAI 배치 파일 생성 중... (300개 기사)
✅ OpenAI 배치 파일 생성 완료: batch_openai.jsonl

📝 Claude 배치 파일 생성 중... (300개 기사)
✅ Claude 배치 파일 생성 완료: batch_claude.jsonl

📤 OpenAI 배치 업로드 중...
✅ 파일 업로드 완료: file-abc123
✅ OpenAI 배치 작업 생성 완료
   배치 ID: batch_xyz789
   상태: validating
   완료 예정: 24시간 이내

📤 Claude 배치 업로드 중...
✅ Claude 배치 작업 생성 완료
   배치 ID: msgbatch_456def
   상태: in_progress
   완료 예정: 24시간 이내

✅ 배치 제출 완료!
```

**배치 ID가 `batch_info.json`에 자동 저장됩니다.**

---

### Step 2: 상태 확인 (몇 시간 후)

배치 작업이 진행 중인지 확인합니다.

```bash
python scripts/batch_labeling.py \
  --mode check \
  --openai-batch-id batch_xyz789 \
  --claude-batch-id msgbatch_456def
```

**출력:**
```
📊 [OpenAI] 배치 상태: in_progress
   총 요청: 300
   완료: 150
   실패: 0

📊 [Claude] 배치 상태: in_progress
   총 요청: 300
   완료: 180
   실패: 0
```

**상태 종류:**
- `validating`: 검증 중
- `in_progress`: 처리 중
- `completed` / `ended`: 완료 ✅
- `failed`: 실패

---

### Step 3: 결과 다운로드 (완료 후)

배치가 완료되면 결과를 다운로드하고 비교합니다.

```bash
python scripts/batch_labeling.py \
  --mode download \
  --input data/unlabeled_news.csv \
  --output data/batch_labeled.csv \
  --openai-batch-id batch_xyz789 \
  --claude-batch-id msgbatch_456def
```

**출력:**
```
⬇️ OpenAI 결과 다운로드 중...
✅ OpenAI 결과 다운로드 완료: 300개

⬇️ Claude 결과 다운로드 중...
✅ Claude 결과 다운로드 완료: 300개

🔍 결과 비교 중...

📊 비교 결과:
   ✅ 일치: 240개 (80.0%)
   ⚠️ 불일치: 60개 (20.0%)

💾 결과 저장 중...
✅ 결과 저장 완료: data/batch_labeled.csv
⚠️ 검토 필요 항목 저장: data/batch_labeled_review_needed.csv (60개)

🎉 라벨링 완료!
```

---

## 📊 출력 파일

### 1. `data/batch_labeled.csv` (전체 결과)

```csv
text,title,source,date,topic,label_openai,label_claude,label_final,agreement,needs_review,review_reason
"기사1...",제목1,조선일보,2024-01-15,부동산,0,0,0,True,False,일치
"기사2...",제목2,한겨레,2024-01-15,경제,1,2,,False,True,불일치
"기사3...",제목3,중앙일보,2024-01-15,외교,2,2,2,True,False,일치
```

**컬럼 설명:**
- `label_openai`: OpenAI 모델 라벨
- `label_claude`: Claude 모델 라벨
- `label_final`: 최종 라벨 (일치 시에만 자동 설정)
- `agreement`: 두 모델 일치 여부
- `needs_review`: 검토 필요 여부
- `review_reason`: 검토 이유 (일치/불일치/API 오류)

### 2. `data/batch_labeled_review_needed.csv` (검토 필요)

두 모델이 불일치한 기사만 포함됩니다.

---

## 💡 팁 & 주의사항

### 배치 작업 시간

- **일반적**: 2-8시간
- **최대**: 24시간
- **빠른 경우**: 1시간 이내

### 비용 절감

```python
# 기본 설정 (권장)
OpenAI: gpt-4o-mini  # $0.015 / 1M input tokens (Batch)
Claude: claude-haiku  # $0.125 / 1M input tokens (Batch)

# 더 정확하게 (2배 비용)
OpenAI: gpt-4o       # $0.625 / 1M input tokens (Batch)
Claude: claude-sonnet # $1.50 / 1M input tokens (Batch)
```

### 상태 확인 주기

```bash
# 1시간마다 확인 (추천)
watch -n 3600 python scripts/batch_labeling.py --mode check ...

# 또는 cron 설정
*/60 * * * * python /path/to/scripts/batch_labeling.py --mode check ...
```

### 에러 처리

배치 작업 중 일부 실패 시:
1. `needs_review=True`로 표시됨
2. `review_reason`에 오류 이유
3. 수동으로 재라벨링 필요

---

## 🔄 검토 필요 항목 처리

### 옵션 1: 수동 검토

```bash
# Excel 또는 Google Sheets로 열기
open data/batch_labeled_review_needed.csv

# label_final 컬럼에 수동 입력
# 0: 옹호, 1: 중립, 2: 비판
```

### 옵션 2: 강력한 모델로 재라벨링

불일치한 항목만 실시간 API로 재검증:

```python
# scripts/review_disagreements.py 사용 (별도 스크립트)
python scripts/review_disagreements.py \
  --input data/batch_labeled_review_needed.csv \
  --output data/final_labeled.csv
```

### 옵션 3: 투표 방식

3개 이상 모델 사용 시 다수결:

```python
# 4개 모델 사용
labels = [0, 1, 0, 0]  # OpenAI-mini, Claude-haiku, GPT-4o, Sonnet
final_label = max(set(labels), key=labels.count)  # 0 (다수)
```

---

## 📈 일치율 개선 팁

### 프롬프트 개선

더 구체적인 기준 제시:

```python
"""
스탠스 분류 기준:

0 (옹호):
  - 정책/인물을 긍정적으로 평가
  - "효과적", "성공적", "바람직" 등의 표현
  - 찬성 입장 강조

1 (중립):
  - 객관적 사실만 전달
  - 찬반 양측 균형있게 제시
  - 평가적 표현 없음

2 (비판):
  - 정책/인물을 부정적으로 평가
  - "문제", "우려", "비판" 등의 표현
  - 반대 입장 강조
"""
```

### Few-shot 예시 추가

```python
messages = [
    {"role": "user", "content": "예시1: [긍정 기사] → 0"},
    {"role": "assistant", "content": "0"},
    {"role": "user", "content": "예시2: [중립 기사] → 1"},
    {"role": "assistant", "content": "1"},
    {"role": "user", "content": "예시3: [부정 기사] → 2"},
    {"role": "assistant", "content": "2"},
    {"role": "user", "content": f"분석할 기사: {article}"},
]
```

---

## 🐛 문제 해결

### "OPENAI_API_KEY not found"

```bash
# .env 파일 확인
cat .env

# 환경 변수 직접 설정
export OPENAI_API_KEY=sk-proj-...
export ANTHROPIC_API_KEY=sk-ant-...
```

### "Batch status: failed"

1. 입력 파일 형식 확인
2. API 키 유효성 확인
3. 크레딧 잔액 확인
4. 에러 로그 확인 (OpenAI Console)

### "No matching distribution found"

```bash
# Python 버전 확인 (3.8 이상 필요)
python --version

# 의존성 재설치
pip install --upgrade -r requirements-scripts.txt
```

---

## 📞 추가 리소스

- **OpenAI Batches 문서**: https://platform.openai.com/docs/guides/batch
- **Claude Batches 문서**: https://docs.anthropic.com/en/docs/build-with-claude/message-batches
- **가격 계산기**: https://openai.com/pricing
- **Claude 가격**: https://www.anthropic.com/pricing

---

## 📊 예상 비용 계산

### 기사 개수별 비용

| 기사 수 | OpenAI (mini) | Claude (haiku) | **총 비용** |
|---------|---------------|----------------|-------------|
| 100 | $0.006 | $0.006 | **$0.012** |
| 300 | $0.018 | $0.018 | **$0.036** |
| 1,000 | $0.060 | $0.060 | **$0.12** |
| 10,000 | $0.600 | $0.600 | **$1.20** |

*평균 기사 길이 500 토큰 기준

---

**🎉 Batches API로 저렴하고 효율적인 대량 라벨링을 시작하세요!**

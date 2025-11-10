# OpenAI Batches API 라벨링 가이드

OpenAI의 Batches API를 사용하여 뉴스 기사를 **50% 할인된 가격**으로 자동 라벨링하는 방법입니다.

---

## 🚀 빠른 시작

### 1️⃣ 배치 제출

```bash
python scripts/batch_labeling_openai.py \
  --input data/unlabeled_news.csv \
  --mode submit
```

**출력:**
```
📝 배치 파일 생성 중... (300개 기사)
✅ 배치 파일 생성 완료: batch_openai.jsonl

📤 배치 업로드 중...
✅ 파일 업로드 완료: file-abc123

✅ 배치 작업 생성 완료!
   배치 ID: batch_xyz789
   상태: validating
   완료 예정: 24시간 이내

✅ 배치 정보 저장: batch_info_openai.json
```

**배치 ID를 기억하세요!** (또는 `batch_info_openai.json` 파일 참고)

---

### 2️⃣ 상태 확인 (몇 시간 후)

```bash
python scripts/batch_labeling_openai.py \
  --mode check \
  --batch-id batch_xyz789
```

**출력:**
```
📊 배치 상태
   ID: batch_xyz789
   상태: in_progress
   진행률: 150/300 (50.0%)
   ⏳ 처리 중... 나중에 다시 확인하세요
```

**상태 종류:**
- `validating`: 검증 중
- `in_progress`: 처리 중 ⏳
- `completed`: 완료 ✅
- `failed`: 실패 ❌

---

### 3️⃣ 결과 다운로드 (완료 후)

```bash
python scripts/batch_labeling_openai.py \
  --mode download \
  --batch-id batch_xyz789 \
  --input data/unlabeled_news.csv \
  --output data/labeled_news.csv
```

**출력:**
```
📂 원본 데이터 로드: data/unlabeled_news.csv
✅ 300개 기사 로드

⬇️ 결과 다운로드 중...
✅ 결과 다운로드 완료
   성공: 298개
   실패: 2개

💾 결과 저장 중...
✅ 결과 저장 완료: data/labeled_news.csv

📊 라벨 분포:
   0 (옹호 (Support)): 95개 (31.7%)
   1 (중립 (Neutral)): 103개 (34.3%)
   2 (비판 (Oppose)): 100개 (33.3%)

⚠️ 라벨링 실패 항목: data/labeled_news_failed.csv (2개)

🎉 라벨링 완료!
```

---

## 📊 입력/출력 파일 형식

### 입력 파일 (CSV)

최소 `text` 컬럼이 필요합니다:

```csv
text,title,source,date,topic
"정부의 새로운 정책은...",기사 제목1,조선일보,2024-01-15,부동산정책
"야당은 이번 법안을...",기사 제목2,한겨레,2024-01-15,경제정책
```

### 출력 파일 (CSV)

라벨과 메타데이터가 추가됩니다:

```csv
text,title,source,date,topic,label,label_name,labeled,labeled_at
"정부의 새로운 정책은...",기사 제목1,조선일보,2024-01-15,부동산정책,0,옹호 (Support),True,2024-01-15 14:30:00
"야당은 이번 법안을...",기사 제목2,한겨레,2024-01-15,경제정책,2,비판 (Oppose),True,2024-01-15 14:30:00
```

**컬럼 설명:**
- `label`: 라벨 (0=옹호, 1=중립, 2=비판)
- `label_name`: 라벨 이름
- `labeled`: 라벨링 성공 여부
- `labeled_at`: 라벨링 시간

---

## 💰 비용

### GPT-4o-mini Batches 가격

| 항목 | 일반 API | **Batches API** | 할인 |
|------|----------|----------------|------|
| Input (1M tokens) | $0.150 | **$0.075** | **50%** |
| Output (1M tokens) | $0.600 | **$0.300** | **50%** |

### 예상 비용 계산

**기사 300개 라벨링:**
```
평균 기사 길이: 500 토큰
프롬프트 길이: 200 토큰
응답 길이: 5 토큰

총 입력: 300 × (500 + 200) = 210,000 tokens
총 출력: 300 × 5 = 1,500 tokens

비용 = (0.21M × $0.075) + (0.0015M × $0.300)
     = $0.016 + $0.0005
     = $0.0165 ≈ $0.02
```

**일반 API 대비 50% 절감!** 💰

---

## ⏱️ 처리 시간

| 기사 수 | 예상 시간 |
|---------|-----------|
| 100 | 1-3시간 |
| 300 | 2-6시간 |
| 1,000 | 4-12시간 |
| 10,000 | 8-24시간 |

**최대 완료 시간**: 24시간

---

## 🔍 배치 ID 확인

### batch_info_openai.json 파일

배치 제출 시 자동 생성됩니다:

```json
{
  "batch_id": "batch_xyz789",
  "submitted_at": "2024-01-15T10:30:00",
  "input_file": "data/unlabeled_news.csv",
  "num_articles": 300
}
```

### 분실 시 복구

OpenAI 대시보드에서 확인:
```
https://platform.openai.com/batches
```

---

## 💡 자동화 팁

### Cron으로 자동 상태 확인

```bash
# 1시간마다 상태 확인
*/60 * * * * cd /path/to/project && python scripts/batch_labeling_openai.py --mode check --batch-id batch_xyz789
```

### 완료 시 자동 다운로드 스크립트

```bash
#!/bin/bash
# auto_download.sh

BATCH_ID="batch_xyz789"

while true; do
  STATUS=$(python scripts/batch_labeling_openai.py --mode check --batch-id $BATCH_ID | grep "상태:" | awk '{print $2}')

  if [ "$STATUS" = "completed" ]; then
    echo "✅ 완료! 결과 다운로드 시작..."
    python scripts/batch_labeling_openai.py \
      --mode download \
      --batch-id $BATCH_ID \
      --input data/unlabeled_news.csv \
      --output data/labeled_news.csv
    break
  fi

  echo "⏳ 처리 중... 1시간 후 재확인"
  sleep 3600
done
```

---

## 🐛 문제 해결

### "OPENAI_API_KEY not found"

```bash
# .env 파일 확인
cat .env | grep OPENAI_API_KEY

# 환경 변수 직접 설정
export OPENAI_API_KEY=sk-proj-...
```

### "Batch status: failed"

**원인:**
1. 입력 파일 형식 오류
2. API 크레딧 부족
3. 잘못된 요청 형식

**해결:**
1. CSV 파일 형식 확인 (UTF-8, text 컬럼 존재)
2. OpenAI 대시보드에서 크레딧 확인
3. batch_openai.jsonl 파일 검사

### "No module named 'openai'"

```bash
# 의존성 설치
pip install -r requirements-scripts.txt

# 또는 직접 설치
pip install openai>=1.30.0
```

### 라벨링 실패 항목 처리

`data/labeled_news_failed.csv` 파일 확인:

```bash
# 실패한 기사만 재라벨링
python scripts/batch_labeling_openai.py \
  --input data/labeled_news_failed.csv \
  --mode submit
```

---

## 📈 대량 데이터 처리

### 10,000개 이상 기사

1. **파일 분할**
```bash
# 1,000개씩 분할
split -l 1000 data/large_dataset.csv data/chunk_
```

2. **배치 제출**
```bash
for file in data/chunk_*; do
  python scripts/batch_labeling_openai.py --input $file --mode submit
  sleep 60  # API rate limit 방지
done
```

3. **결과 병합**
```bash
cat data/labeled_chunk_* > data/labeled_all.csv
```

---

## 🎯 모범 사례

### 1. 프롬프트 개선

더 정확한 분류를 위해 프롬프트 커스터마이징:

```python
# scripts/batch_labeling_openai.py 수정

content = f"""다음 뉴스 기사의 논조를 분석하세요.

제목: {row.get('title', 'N/A')}
언론사: {row.get('source', 'N/A')}
날짜: {row.get('date', 'N/A')}

기사 내용:
{row['text']}

위 기사의 스탠스를 다음 기준으로 분류하세요:

0 (옹호):
  - 정책/인물을 긍정적으로 평가
  - "효과적", "성공적" 등 긍정 표현
  - 찬성 입장 강조

1 (중립):
  - 객관적 사실만 전달
  - 찬반 양측 균형 제시
  - 평가적 표현 최소

2 (비판):
  - 정책/인물을 부정적으로 평가
  - "문제", "우려" 등 부정 표현
  - 반대 입장 강조

숫자(0, 1, 2) 하나만 출력하세요."""
```

### 2. Few-shot 예시 추가

```python
messages = [
    {"role": "system", "content": "당신은 정치 뉴스 분석 전문가입니다."},
    {"role": "user", "content": "예시: [긍정적 기사] → 0"},
    {"role": "assistant", "content": "0"},
    {"role": "user", "content": "예시: [중립적 기사] → 1"},
    {"role": "assistant", "content": "1"},
    {"role": "user", "content": "예시: [부정적 기사] → 2"},
    {"role": "assistant", "content": "2"},
    {"role": "user", "content": f"분석할 기사:\n{row['text']}"},
]
```

### 3. Temperature 조정

```python
"temperature": 0,  # 일관성 (추천)
"temperature": 0.3,  # 약간의 변동성
```

---

## 📞 추가 리소스

- **OpenAI Batches 문서**: https://platform.openai.com/docs/guides/batch
- **API 사용량 확인**: https://platform.openai.com/usage
- **요금제**: https://openai.com/pricing
- **상태 대시보드**: https://platform.openai.com/batches

---

**🎉 OpenAI Batches API로 효율적인 대량 라벨링을 시작하세요!**

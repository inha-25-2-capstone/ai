# 데이터 수집 및 검증 스크립트

이 디렉토리에는 뉴스 데이터 수집과 검증을 위한 유틸리티 스크립트가 포함되어 있습니다.

## 🚀 설치

스크립트 실행 전, 필요한 의존성을 설치하세요:

```bash
# 자동 라벨링 스크립트용 의존성 (LLM API 포함)
pip install -r requirements-scripts.txt
```

**참고**: `auto_labeling.py`와 `review_labels.py`는 OpenAI와 Google Gemini API를 사용합니다.
다른 스크립트는 기본 `requirements.txt`만으로 실행 가능합니다.

## 📁 파일 목록

### 1. collect_news.py
뉴스 기사를 수집하는 스크립트

**기능:**
- 샘플 데이터 생성 (테스트용)
- Naver News API 연동 (실제 데이터 수집)
- CSV/JSON 형식으로 저장

**사용법:**

```bash
# 샘플 데이터 생성 (30개)
python scripts/collect_news.py --query "부동산 정책" --num 30 --mode sample

# Naver API로 실제 데이터 수집
python scripts/collect_news.py \
    --query "부동산 정책" \
    --num 100 \
    --mode naver \
    --naver-id YOUR_CLIENT_ID \
    --naver-secret YOUR_CLIENT_SECRET

# JSON 형식으로 저장
python scripts/collect_news.py \
    --query "경제 정책" \
    --num 50 \
    --output data/news.json
```

**출력 형식:**
```csv
text,label,source,date,topic,url,note
"기사 본문...",None,조선일보,2024-01-15,부동산정책,https://...,unlabeled
```

---

### 2. validate_data.py
라벨링된 데이터의 품질을 검증하는 스크립트

**검증 항목:**
- 필수 컬럼 확인 (text, label)
- 레이블 유효성 검증 (0, 1, 2만 허용)
- 텍스트 품질 확인 (길이, 결측치 등)
- 클래스 균형 확인
- 중복 데이터 확인

**사용법:**

```bash
# 데이터 검증
python scripts/validate_data.py --input data/labeled_news.csv
```

**출력 예시:**
```
📊 데이터 검증 시작
============================================================

📋 기본 정보
   전체 샘플 수: 300개
   컬럼: text, label, source, date, topic

🏷️  레이블 검증
   ✅ 모든 레이블 값이 유효합니다 (0, 1, 2)

   레이블 분포:
      0 (옹호): 100개 (33.3%)
      1 (중립): 100개 (33.3%)
      2 (비판): 100개 (33.3%)

📝 텍스트 품질 검증
   텍스트 길이 통계:
      평균: 250자
      최소: 120자
      최대: 800자

⚖️  클래스 균형 확인
   ✅ 클래스 균형이 양호합니다

🔄 중복 확인
   ✅ 중복 없음

============================================================
✅ 모든 검증 통과! 데이터가 학습에 적합합니다.
============================================================
```

---

## 🔄 워크플로우

### 1단계: 데이터 수집

```bash
# 샘플 데이터로 시작 (테스트용)
python scripts/collect_news.py \
    --query "부동산 정책" \
    --num 30 \
    --mode sample \
    --output data/unlabeled_news.csv
```

### 2단계: 라벨링

Excel 또는 Google Sheets에서 `data/unlabeled_news.csv` 파일을 열고 `label` 컬럼에 값 입력:
- 0: 옹호 (Support)
- 1: 중립 (Neutral)
- 2: 비판 (Oppose)

라벨링 완료 후 `data/labeled_news.csv`로 저장

### 3단계: 데이터 검증

```bash
python scripts/validate_data.py --input data/labeled_news.csv
```

검증 통과하면 다음 단계로 진행

### 4단계: 모델 학습

Colab 노트북(`notebooks/stance_training.ipynb`)에서:
1. `USE_REAL_DATA = True`로 변경
2. `data/labeled_news.csv` 업로드
3. 노트북 실행

---

## 📚 추가 리소스

- **데이터 형식 가이드**: `data/DATA_PREPARATION_GUIDE.md`
- **수집 가이드**: `data/NEWS_DATA_COLLECTION_GUIDE.md`
- **학습 노트북**: `notebooks/stance_training.ipynb`

---

## 💡 팁

### Naver API 키 발급
1. https://developers.naver.com/apps/ 접속
2. 애플리케이션 등록
3. "검색" API 선택
4. Client ID와 Client Secret 복사

### 환경 변수 설정 (선택사항)
API 키를 매번 입력하지 않으려면:

```bash
# Windows
set NAVER_CLIENT_ID=your_id
set NAVER_CLIENT_SECRET=your_secret

# Mac/Linux
export NAVER_CLIENT_ID=your_id
export NAVER_CLIENT_SECRET=your_secret
```

스크립트 수정:
```python
import os
client_id = os.getenv('NAVER_CLIENT_ID')
client_secret = os.getenv('NAVER_CLIENT_SECRET')
```

---

## 🐛 문제 해결

### "ModuleNotFoundError: No module named 'pandas'"
```bash
pip install pandas
```

### "requests 패키지가 필요합니다"
```bash
pip install requests
```

### API 오류 (403, 429 등)
- Client ID/Secret 확인
- API 호출 제한 확인 (하루 25,000건)
- 시간 간격을 두고 재시도

---

## 📞 문의

스크립트 사용 중 문제가 있으면 GitHub Issues에 올려주세요.

# 뉴스 데이터 수집 및 라벨링 가이드

실제 뉴스 기사를 수집하고 스탠스 분류 모델 학습을 위한 데이터를 준비하는 방법을 안내합니다.

---

## 📊 목표 데이터 규모

### 최소 요구 사항
- **클래스당**: 100개 이상
- **전체**: 300개 이상
- **권장**: 클래스당 300-500개 (전체 1,000-1,500개)

### 단계별 목표
| 단계 | 클래스당 개수 | 전체 개수 | 용도 |
|------|--------------|----------|------|
| 1단계 (베타) | 100개 | 300개 | 초기 모델 학습 및 검증 |
| 2단계 (개선) | 300개 | 900개 | 성능 개선 |
| 3단계 (프로덕션) | 1,000개 | 3,000개 | 실전 배포 |

---

## 🎯 1. 데이터 수집 전략

### 1.1 토픽 선정
**정치 이슈 중심**으로 수집 (프로젝트 특성상)

#### 추천 토픽 예시:
- 경제 정책 (부동산 정책, 세금 정책 등)
- 노동 정책 (최저임금, 근로시간 등)
- 외교 정책 (한미관계, 한일관계 등)
- 사회 정책 (교육개혁, 의료개혁 등)
- 환경 정책 (탄소중립, 원전 정책 등)

**중요:** 같은 이슈에 대해 다양한 논조의 기사를 수집해야 합니다.

### 1.2 언론사 선정
**다양한 논조**의 언론사를 포함해야 합니다.

#### 논조별 언론사 예시:
- **보수 성향**: 조선일보, 중앙일보, 동아일보
- **진보 성향**: 한겨레, 경향신문, 한국일보
- **중도/경제**: 매일경제, 한국경제, 서울경제
- **공영**: 연합뉴스, KBS, MBC

**목표:** 각 토픽마다 최소 3-5개 언론사에서 수집

---

## 🔧 2. 데이터 수집 방법

### 방법 A: 뉴스 API 활용 (추천)

#### 2.1 Naver News API
```python
import requests
import pandas as pd

# Naver API 키 필요 (https://developers.naver.com/apps/)
CLIENT_ID = "your_client_id"
CLIENT_SECRET = "your_client_secret"

def search_news(query, display=100):
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": CLIENT_ID,
        "X-Naver-Client-Secret": CLIENT_SECRET
    }
    params = {
        "query": query,
        "display": display,
        "sort": "date"
    }

    response = requests.get(url, headers=headers, params=params)
    return response.json()

# 사용 예시
results = search_news("부동산 정책", display=100)
articles = results['items']

# DataFrame으로 변환
df = pd.DataFrame(articles)
df.to_csv('news_data.csv', index=False, encoding='utf-8-sig')
```

#### 2.2 Google News (무료, API 키 불필요)
```python
from gnews import GNews

google_news = GNews(language='ko', country='KR', max_results=100)
articles = google_news.get_news('부동산 정책')

for article in articles:
    print(article['title'])
    print(article['url'])
```

### 방법 B: 웹 크롤링 (고급)

**주의:** 각 언론사의 robots.txt 확인 및 이용약관 준수 필요

```python
import requests
from bs4 import BeautifulSoup
import time

def crawl_news_article(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # 언론사마다 구조가 다르므로 조정 필요
        title = soup.find('h1').text.strip()
        content = soup.find('article').text.strip()

        return {
            'title': title,
            'content': content,
            'url': url
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

# 예시
article = crawl_news_article('https://example.com/news/123')
```

### 방법 C: 수동 수집 (소규모)

1. **언론사 웹사이트 방문**
2. **특정 토픽 검색**
3. **기사 URL 및 본문 복사**
4. **스프레드시트에 정리**

---

## 🏷️ 3. 라벨링 가이드

### 3.1 라벨링 원칙

#### 스탠스 정의
| 레이블 | 코드 | 정의 | 특징 |
|--------|------|------|------|
| **옹호 (Support)** | 0 | 특정 이슈/정책/인물을 긍정적으로 평가 | 긍정 표현, 장점 강조, 지지 입장 |
| **중립 (Neutral)** | 1 | 사실 전달 위주, 특정 입장 없음 | 객관적 서술, 양측 의견 균형있게 제시 |
| **비판 (Oppose)** | 2 | 특정 이슈/정책/인물을 부정적으로 평가 | 부정 표현, 문제점 강조, 반대 입장 |

### 3.2 라벨링 절차

#### Step 1: 기사 전체 읽기
- 제목만 보고 판단하지 말 것
- 전체 맥락 파악

#### Step 2: 핵심 메시지 파악
- 기사의 주요 논조는?
- 기자의 의도는?

#### Step 3: 레이블 결정
- 명확한 경우 즉시 분류
- 애매한 경우 "중립"로 분류

#### Step 4: 검증
- 2명 이상이 독립적으로 라벨링
- 불일치 시 토론 후 결정

### 3.3 라벨링 예시

#### ✅ 옹호 (Support) 예시
```
"이번 부동산 정책은 실수요자 보호에 큰 도움이 될 것으로 보인다.
전문가들은 집값 안정화에 긍정적 효과를 기대하고 있다."
→ 레이블: 0
```

#### ✅ 중립 (Neutral) 예시
```
"정부가 오늘 새로운 부동산 정책을 발표했다.
주요 내용은 대출 규제 강화와 공급 확대다.
업계와 시민단체의 반응은 엇갈리고 있다."
→ 레이블: 1
```

#### ✅ 비판 (Oppose) 예시
```
"이번 부동산 정책은 실효성이 의심된다.
전문가들은 시장 혼란만 가중시킬 것이라고 우려했다."
→ 레이블: 2
```

### 3.4 애매한 케이스 처리

| 상황 | 처리 방법 |
|------|----------|
| 긍정과 부정이 섞여있음 | 전체적인 논조의 무게 중심으로 판단 |
| 사실만 나열, 판단 불가 | 중립(1)로 분류 |
| 간접 인용만 있음 | 기사 전체 맥락으로 판단 |
| 풍자/비꼬기 | 실제 의도를 파악하여 분류 |

---

## 📝 4. 데이터 정리 및 저장

### 4.1 필수 컬럼
```csv
text,label,source,date,topic
"기사 본문...",0,"조선일보","2024-01-15","부동산정책"
"기사 본문...",1,"연합뉴스","2024-01-15","부동산정책"
```

### 4.2 Excel/Google Sheets 템플릿

| text | label | source | date | topic | url | note |
|------|-------|--------|------|-------|-----|------|
| 기사 본문 | 0 | 조선일보 | 2024-01-15 | 부동산정책 | https://... | 검토 완료 |

### 4.3 데이터 검증 체크리스트

#### 전처리
- [ ] 중복 제거
- [ ] HTML 태그 제거
- [ ] 특수문자 정리
- [ ] 광고/관련기사 문구 제거

#### 품질 검증
- [ ] 너무 짧은 기사 제외 (100자 미만)
- [ ] 기사가 아닌 내용 제외 (광고, 공지사항 등)
- [ ] 라벨이 명확하지 않은 샘플 제외 또는 재검토

#### 균형 확인
- [ ] 클래스별 비율 확인 (최대/최소 비율 3:1 이하 권장)
- [ ] 토픽별 분포 확인
- [ ] 언론사별 분포 확인

---

## 🛠️ 5. 데이터 수집 도구 (Python Script)

### 완전한 수집 스크립트
```python
import pandas as pd
import requests
from datetime import datetime
import time

class NewsCollector:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.articles = []

    def collect_from_naver(self, query, num_articles=100):
        """Naver News API로 수집"""
        # API 호출 로직
        pass

    def save_to_csv(self, filename='collected_news.csv'):
        """CSV로 저장"""
        df = pd.DataFrame(self.articles)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ {len(self.articles)}개 기사 저장 완료: {filename}")

    def add_manual_article(self, text, source, date, topic, url=''):
        """수동으로 기사 추가"""
        self.articles.append({
            'text': text,
            'label': None,  # 나중에 라벨링
            'source': source,
            'date': date,
            'topic': topic,
            'url': url
        })

# 사용 예시
collector = NewsCollector()
collector.collect_from_naver("부동산 정책", num_articles=100)
collector.save_to_csv('news_unlabeled.csv')
```

---

## 📊 6. 라벨링 도구 추천

### 도구 A: Label Studio (추천)
- **장점**: 웹 기반, 협업 가능, 무료
- **설치**: `pip install label-studio`
- **실행**: `label-studio start`
- **URL**: http://localhost:8080

### 도구 B: Google Sheets + Scripts
- **장점**: 간단, 접근성 좋음
- **단점**: 대량 데이터 처리 어려움

### 도구 C: Excel + VBA
- **장점**: 오프라인 작업 가능
- **단점**: 협업 어려움

---

## 🚀 7. 실전 워크플로우

### Week 1: 데이터 수집
```
Day 1-2: 토픽 선정 및 언론사 리스트 작성
Day 3-5: API/크롤링으로 기사 수집 (목표: 500-1000개)
Day 6-7: 중복 제거 및 전처리
```

### Week 2-3: 라벨링
```
Day 1: 라벨링 가이드라인 확정 및 팀원 교육
Day 2-10: 라벨링 작업 (목표: 하루 50개 × 2명 = 100개)
Day 11-14: 불일치 샘플 재검토 및 품질 검증
```

### Week 4: 데이터 정리 및 학습
```
Day 1-2: 최종 데이터 정리 및 형식 변환
Day 3-5: 모델 학습 및 평가
Day 6-7: 성능 분석 및 개선
```

---

## 💡 8. 팁과 주의사항

### 라벨링 팁
1. **일관성 유지**: 라벨링 가이드를 항상 참고
2. **휴식**: 하루 100개 이상 하지 말 것 (피로도 ↑, 품질 ↓)
3. **불확실하면 표시**: 나중에 재검토
4. **Inter-Annotator Agreement**: 팀원 간 일치도 정기 확인

### 주의사항
1. **저작권**: 기사 원문 사용 시 저작권 확인
2. **개인정보**: 개인정보 포함된 기사 제외
3. **윤리**: 특정 정치 성향으로 편향되지 않도록 주의
4. **백업**: 정기적으로 백업 (Google Drive, GitHub 등)

---

## 📞 9. 도움이 필요할 때

### 자동화 도구가 필요하면
- 크롤링 스크립트 작성 지원
- API 연동 코드 제공

### 라벨링 중 어려움이 있으면
- 애매한 케이스 판단 기준 제공
- 라벨링 품질 검증 도구 제공

### 데이터 전처리가 필요하면
- 텍스트 클리닝 스크립트 제공
- 데이터 검증 스크립트 제공

---

## ✅ 체크리스트

### 시작 전
- [ ] 토픽 리스트 작성
- [ ] 언론사 리스트 작성
- [ ] 수집 도구 준비 (API 키 또는 크롤러)
- [ ] 라벨링 가이드 숙지

### 수집 중
- [ ] 일일 진행 상황 기록
- [ ] 중복 제거
- [ ] 품질 체크 (기사가 너무 짧거나 내용 없음)

### 라벨링 중
- [ ] 라벨링 가이드 준수
- [ ] 애매한 샘플 별도 표시
- [ ] 팀원 간 일치도 확인

### 완료 후
- [ ] 최종 데이터 개수 확인 (목표 달성 여부)
- [ ] 클래스 균형 확인
- [ ] CSV/JSON 형식으로 저장
- [ ] Colab 노트북에 업로드하여 학습

---

## 📁 참고 자료

- `DATA_PREPARATION_GUIDE.md`: 데이터 형식 상세 가이드
- `stance_training.ipynb`: 학습 노트북
- `saved_models/`: 학습된 모델 저장 위치

---

**목표**: 클래스당 최소 100개 이상 수집하여 모델 성능 70% 이상 달성! 🎯

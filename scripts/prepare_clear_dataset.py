"""
명확한 스탠스 데이터셋 생성 스크립트

기존 train.csv + 새로운 JSON 데이터에서 명확한 옹호/비판/중립 샘플만 추출하여
하나의 학습 데이터셋을 생성합니다.

Usage:
    python scripts/prepare_clear_dataset.py
"""

import pandas as pd
import json
import re
import os
import sys
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# 설정
# ============================================================================

# 파일 경로
TRAIN_CSV_PATH = r"c:\Users\gaaahee\Downloads\train.csv"
JSON_PATH = r"c:\Users\gaaahee\OneDrive\문서\카카오톡 받은 파일\bigkinds_1년치_summarized.json"
OUTPUT_DIR = "data/clear_dataset"

# 키워드 정의 (확장)
SUPPORT_KEYWORDS = [
    '기대', '성과', '효과', '발전', '성공', '긍정', '환영', '지지', '칭찬', '훌륭',
    '기여', '도움', '성장', '개선', '향상', '진전', '협력', '합의', '약속', '의지',
    '노력', '추진', '강조', '확대', '강화', '증가', '상승', '호평', '찬사', '격려'
]

OPPOSE_KEYWORDS = [
    '비판', '규탄', '반대', '우려', '문제', '실패', '졸속', '부실', '논란', '항의',
    '질타', '맹비난', '지적', '비난', '불만', '갈등', '충돌', '위기', '혼란', '파행',
    '거부', '철회', '중단', '취소', '폐기', '반발', '저항', '시위', '탄핵', '파면'
]

# 필터링 기준
SUPPORT_THRESHOLD = 0.70  # 옹호 키워드 비율 70% 이상
OPPOSE_THRESHOLD = 0.30   # 옹호 키워드 비율 30% 이하 (= 비판 70% 이상)
NEUTRAL_LOW = 0.40        # 중립: 40% ~ 60%
NEUTRAL_HIGH = 0.60


# ============================================================================
# 유틸리티 함수
# ============================================================================

def calculate_stance_score(text):
    """텍스트의 스탠스 점수 계산 (0=비판, 0.5=중립, 1=옹호)"""
    text = str(text)
    support_count = sum(text.count(kw) for kw in SUPPORT_KEYWORDS)
    oppose_count = sum(text.count(kw) for kw in OPPOSE_KEYWORDS)
    total = support_count + oppose_count
    if total == 0:
        return 0.5  # 키워드 없으면 중립
    return support_count / total


def assign_label_by_keywords(text):
    """키워드 기반으로 라벨 할당 (명확하지 않으면 None)"""
    score = calculate_stance_score(text)
    if score >= SUPPORT_THRESHOLD:
        return 0  # 옹호
    elif score <= OPPOSE_THRESHOLD:
        return 2  # 비판
    elif NEUTRAL_LOW <= score <= NEUTRAL_HIGH:
        return 1  # 중립
    else:
        return None  # 애매함 - 제외


def load_json_data(filepath):
    """JSON 파일 로드 (trailing comma 처리)"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # trailing comma 제거
    content = re.sub(r',\s*}', '}', content)
    content = re.sub(r',\s*]', ']', content)

    return json.loads(content)


def clean_text(text):
    """텍스트 정리 (개행문자 등 제거)"""
    if pd.isna(text):
        return ""
    text = str(text)
    # 연속 개행 제거
    text = re.sub(r'\n+', ' ', text)
    # 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ============================================================================
# 메인 로직
# ============================================================================

def main():
    print("=" * 60)
    print("명확한 스탠스 데이터셋 생성")
    print("=" * 60)

    # 1. 기존 train.csv 로드 및 필터링
    print("\n[1/4] 기존 train.csv 로드 중...")
    train_df = pd.read_csv(TRAIN_CSV_PATH, encoding='utf-8')
    print(f"  - 원본 데이터: {len(train_df)}개")

    # 스탠스 점수 계산
    train_df['stance_score'] = train_df['content'].apply(calculate_stance_score)

    # 명확한 샘플 필터링
    clear_support_train = train_df[(train_df['label'] == 0) & (train_df['stance_score'] >= SUPPORT_THRESHOLD)]
    clear_neutral_train = train_df[(train_df['label'] == 1) &
                                    (train_df['stance_score'] >= NEUTRAL_LOW) &
                                    (train_df['stance_score'] <= NEUTRAL_HIGH)]
    clear_oppose_train = train_df[(train_df['label'] == 2) & (train_df['stance_score'] <= OPPOSE_THRESHOLD)]

    print(f"  - 명확한 옹호: {len(clear_support_train)}개")
    print(f"  - 명확한 중립: {len(clear_neutral_train)}개")
    print(f"  - 명확한 비판: {len(clear_oppose_train)}개")

    # 2. JSON 데이터 로드 및 라벨링
    print("\n[2/4] JSON 데이터 로드 및 키워드 기반 라벨링...")
    json_data = load_json_data(JSON_PATH)
    print(f"  - 원본 데이터: {len(json_data)}개")

    # 부고/인사 등 제외
    exclude_patterns = ['부고', '인사', '부음', '장례']

    json_labeled = []
    for item in json_data:
        title = item.get('title', '')
        summary = item.get('summary', '')

        # 제외 패턴 체크
        if any(p in title for p in exclude_patterns):
            continue

        # 텍스트 결합 (제목 + 요약)
        text = clean_text(title + " " + summary)

        if len(text) < 50:  # 너무 짧은 텍스트 제외
            continue

        # 키워드 기반 라벨 할당
        label = assign_label_by_keywords(text)

        if label is not None:
            json_labeled.append({
                'title': title,
                'content': text,
                'label': label,
                'source': 'json'
            })

    json_df = pd.DataFrame(json_labeled)

    # JSON에서 추출한 클래스별 개수
    json_support = json_df[json_df['label'] == 0]
    json_neutral = json_df[json_df['label'] == 1]
    json_oppose = json_df[json_df['label'] == 2]

    print(f"  - 추출된 옹호: {len(json_support)}개")
    print(f"  - 추출된 중립: {len(json_neutral)}개")
    print(f"  - 추출된 비판: {len(json_oppose)}개")

    # 3. 데이터 합치기
    print("\n[3/4] 데이터 합치기...")

    # train.csv 데이터 정리
    train_clear = pd.concat([clear_support_train, clear_neutral_train, clear_oppose_train])
    train_clear = train_clear[['title', 'content', 'label']].copy()
    train_clear['source'] = 'csv'

    # 합치기
    combined_df = pd.concat([train_clear, json_df], ignore_index=True)

    # 중복 제거 (제목 기준)
    combined_df = combined_df.drop_duplicates(subset=['title'], keep='first')

    print(f"  - 합친 후 총 데이터: {len(combined_df)}개")

    # 클래스별 개수 확인
    support_total = len(combined_df[combined_df['label'] == 0])
    neutral_total = len(combined_df[combined_df['label'] == 1])
    oppose_total = len(combined_df[combined_df['label'] == 2])

    print(f"  - 옹호: {support_total}개")
    print(f"  - 중립: {neutral_total}개")
    print(f"  - 비판: {oppose_total}개")

    # 4. 균형 맞추기 및 분할
    print("\n[4/4] 균형 맞추기 및 train/val/test 분할...")

    # 각 클래스에서 동일 개수 샘플링
    min_count = min(support_total, neutral_total, oppose_total)
    print(f"  - 클래스당 샘플 수: {min_count}개")

    balanced_support = combined_df[combined_df['label'] == 0].sample(n=min_count, random_state=42)
    balanced_neutral = combined_df[combined_df['label'] == 1].sample(n=min_count, random_state=42)
    balanced_oppose = combined_df[combined_df['label'] == 2].sample(n=min_count, random_state=42)

    balanced_df = pd.concat([balanced_support, balanced_neutral, balanced_oppose])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # 셔플

    print(f"  - 균형 맞춘 총 데이터: {len(balanced_df)}개")

    # train/val/test 분할 (80/10/10)
    train_data, temp_data = train_test_split(balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['label'])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['label'])

    print(f"  - Train: {len(train_data)}개")
    print(f"  - Val: {len(val_data)}개")
    print(f"  - Test: {len(test_data)}개")

    # 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_data.to_csv(f"{OUTPUT_DIR}/train_dataset.csv", index=False, encoding='utf-8-sig')
    val_data.to_csv(f"{OUTPUT_DIR}/val_dataset.csv", index=False, encoding='utf-8-sig')
    test_data.to_csv(f"{OUTPUT_DIR}/test_dataset.csv", index=False, encoding='utf-8-sig')

    # 전체 데이터도 저장
    balanced_df.to_csv(f"{OUTPUT_DIR}/full_dataset.csv", index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"\n저장된 파일:")
    print(f"  - {OUTPUT_DIR}/train_dataset.csv ({len(train_data)}개)")
    print(f"  - {OUTPUT_DIR}/val_dataset.csv ({len(val_data)}개)")
    print(f"  - {OUTPUT_DIR}/test_dataset.csv ({len(test_data)}개)")
    print(f"  - {OUTPUT_DIR}/full_dataset.csv ({len(balanced_df)}개)")

    # 최종 통계
    print("\n클래스 분포 (Train):")
    for label, name in [(0, '옹호'), (1, '중립'), (2, '비판')]:
        count = len(train_data[train_data['label'] == label])
        print(f"  {label} ({name}): {count}개 ({count/len(train_data)*100:.1f}%)")


if __name__ == '__main__':
    main()

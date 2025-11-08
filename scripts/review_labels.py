"""
라벨 검수 스크립트

자동 라벨링된 데이터 중 검수가 필요한 샘플을 찾아서 검토합니다.

사용법:
    # 신뢰도 낮은 샘플만 검수
    python scripts/review_labels.py --input data/labeled.csv --output data/reviewed.csv --low-confidence

    # API 불일치 샘플만 검수
    python scripts/review_labels.py --input data/labeled.csv --output data/reviewed.csv --disagreement

    # 무작위 샘플링 검수
    python scripts/review_labels.py --input data/labeled.csv --output data/reviewed.csv --random 100
"""

import argparse
import pandas as pd
import os
import sys
from typing import Optional


class LabelReviewer:
    """라벨 검수 클래스"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.reviewed_indices = []
        self.changes = []

    def filter_low_confidence(self, threshold: float = 0.7) -> pd.DataFrame:
        """신뢰도가 낮은 샘플 필터링"""
        if 'confidence' not in self.df.columns:
            print("⚠️  'confidence' 컬럼이 없습니다.")
            return pd.DataFrame()

        low_conf = self.df[self.df['confidence'] < threshold].copy()
        print(f"\n📊 신뢰도 {threshold} 미만 샘플: {len(low_conf)}개 (전체의 {len(low_conf)/len(self.df)*100:.1f}%)")
        return low_conf

    def filter_disagreement(self) -> pd.DataFrame:
        """API 간 불일치 샘플 필터링"""
        if 'agreement' not in self.df.columns:
            print("⚠️  'agreement' 컬럼이 없습니다.")
            return pd.DataFrame()

        disagreement = self.df[self.df['agreement'] == False].copy()
        print(f"\n📊 API 간 불일치 샘플: {len(disagreement)}개 (전체의 {len(disagreement)/len(self.df)*100:.1f}%)")
        return disagreement

    def random_sample(self, n: int) -> pd.DataFrame:
        """무작위 샘플링"""
        sample = self.df.sample(n=min(n, len(self.df)), random_state=42)
        print(f"\n📊 무작위 샘플: {len(sample)}개")
        return sample

    def review_interactive(self, df_to_review: pd.DataFrame):
        """대화형 검수"""
        if len(df_to_review) == 0:
            print("\n검수할 샘플이 없습니다.")
            return

        print(f"\n{'='*80}")
        print(f"🔍 검수 시작 ({len(df_to_review)}개 샘플)")
        print(f"{'='*80}")
        print("\n명령어:")
        print("  0, 1, 2: 레이블 변경")
        print("  k: 현재 레이블 유지")
        print("  s: 건너뛰기")
        print("  d: 삭제 표시")
        print("  q: 종료")
        print(f"{'='*80}\n")

        label_names = {0: '옹호', 1: '중립', 2: '비판'}

        for idx, row in df_to_review.iterrows():
            print(f"\n{'─'*80}")
            print(f"샘플 {idx + 1}/{len(df_to_review)}")
            print(f"{'─'*80}")

            # 텍스트 출력 (길면 잘라서)
            text = row['text']
            if len(text) > 500:
                print(f"텍스트: {text[:500]}...\n")
            else:
                print(f"텍스트: {text}\n")

            # 현재 레이블 정보
            current_label = int(row['label'])
            print(f"현재 레이블: {current_label} ({label_names[current_label]})")

            if 'confidence' in row:
                print(f"신뢰도: {row['confidence']:.3f}")

            if 'reasons' in row and pd.notna(row['reasons']):
                print(f"근거: {row['reasons']}")

            if 'providers' in row:
                print(f"사용 API: {row['providers']}")

            # 사용자 입력
            while True:
                user_input = input("\n결정 [0/1/2/k/s/d/q]: ").strip().lower()

                if user_input == 'q':
                    print("\n검수를 종료합니다.")
                    return

                if user_input == 's':
                    print("⏭️  건너뛰기")
                    break

                if user_input == 'k':
                    print(f"✅ 현재 레이블 유지: {current_label}")
                    self.reviewed_indices.append(idx)
                    break

                if user_input == 'd':
                    print("🗑️  삭제 표시")
                    self.df.at[idx, 'to_delete'] = True
                    self.changes.append({
                        'index': idx,
                        'old_label': current_label,
                        'new_label': None,
                        'action': 'delete'
                    })
                    break

                if user_input in ['0', '1', '2']:
                    new_label = int(user_input)
                    if new_label != current_label:
                        print(f"✏️  레이블 변경: {current_label} → {new_label}")
                        self.df.at[idx, 'label'] = new_label
                        self.df.at[idx, 'human_reviewed'] = True
                        self.changes.append({
                            'index': idx,
                            'old_label': current_label,
                            'new_label': new_label,
                            'action': 'change'
                        })
                    else:
                        print(f"✅ 레이블 확인: {new_label}")
                        self.df.at[idx, 'human_reviewed'] = True

                    self.reviewed_indices.append(idx)
                    break

                print("❌ 잘못된 입력입니다. 다시 입력하세요.")

        print(f"\n{'='*80}")
        print(f"✅ 검수 완료!")
        print(f"{'='*80}")

    def print_summary(self):
        """검수 결과 요약"""
        print(f"\n📊 검수 요약")
        print(f"{'='*60}")
        print(f"검수한 샘플: {len(self.reviewed_indices)}개")

        if self.changes:
            print(f"\n변경 내역: {len(self.changes)}건")

            change_count = len([c for c in self.changes if c['action'] == 'change'])
            delete_count = len([c for c in self.changes if c['action'] == 'delete'])

            print(f"  레이블 변경: {change_count}건")
            print(f"  삭제 표시: {delete_count}건")

            # 변경 패턴 분석
            if change_count > 0:
                print(f"\n변경 패턴:")
                from collections import Counter
                patterns = Counter()
                for c in self.changes:
                    if c['action'] == 'change':
                        patterns[f"{c['old_label']} → {c['new_label']}"] += 1

                for pattern, count in patterns.most_common():
                    print(f"  {pattern}: {count}건")

        print(f"{'='*60}")

    def save(self, output_path: str):
        """검수 결과 저장"""
        # 삭제 표시된 샘플 제거
        if 'to_delete' in self.df.columns:
            before_count = len(self.df)
            self.df = self.df[self.df['to_delete'] != True]
            deleted_count = before_count - len(self.df)
            if deleted_count > 0:
                print(f"\n🗑️  {deleted_count}개 샘플 삭제됨")

        # 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 저장 완료: {output_path}")

        # 최종 통계
        print(f"\n최종 데이터셋 정보:")
        print(f"  전체 샘플: {len(self.df)}개")

        label_names = {0: '옹호', 1: '중립', 2: '비판'}
        print(f"\n  레이블 분포:")
        for label in [0, 1, 2]:
            count = (self.df['label'] == label).sum()
            percentage = count / len(self.df) * 100 if len(self.df) > 0 else 0
            print(f"    {label} ({label_names[label]}): {count}개 ({percentage:.1f}%)")

        if 'human_reviewed' in self.df.columns:
            reviewed_count = self.df['human_reviewed'].sum()
            print(f"\n  사람이 검수한 샘플: {reviewed_count}개 ({reviewed_count/len(self.df)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='라벨 검수')
    parser.add_argument('--input', type=str, required=True, help='입력 CSV 파일')
    parser.add_argument('--output', type=str, required=True, help='출력 CSV 파일')
    parser.add_argument('--low-confidence', action='store_true', help='신뢰도 낮은 샘플만 검수')
    parser.add_argument('--confidence-threshold', type=float, default=0.7, help='신뢰도 임계값')
    parser.add_argument('--disagreement', action='store_true', help='API 불일치 샘플만 검수')
    parser.add_argument('--random', type=int, help='무작위 N개 샘플 검수')
    parser.add_argument('--all', action='store_true', help='전체 샘플 검수')

    args = parser.parse_args()

    # 데이터 로드
    print(f"\n📂 데이터 로딩: {args.input}")
    df = pd.read_csv(args.input)
    print(f"전체 샘플 수: {len(df)}개")

    reviewer = LabelReviewer(df)

    # 검수 대상 필터링
    if args.low_confidence:
        df_to_review = reviewer.filter_low_confidence(args.confidence_threshold)
    elif args.disagreement:
        df_to_review = reviewer.filter_disagreement()
    elif args.random:
        df_to_review = reviewer.random_sample(args.random)
    elif args.all:
        df_to_review = df
        print(f"\n📊 전체 샘플 검수: {len(df_to_review)}개")
    else:
        print("\n⚠️  검수 모드를 선택하세요: --low-confidence, --disagreement, --random N, 또는 --all")
        return

    if len(df_to_review) == 0:
        print("\n검수할 샘플이 없습니다.")
        return

    # 대화형 검수
    reviewer.review_interactive(df_to_review)

    # 결과 요약
    reviewer.print_summary()

    # 저장
    reviewer.save(args.output)

    print(f"\n다음 단계:")
    print(f"1. 데이터 검증: python scripts/validate_data.py --input {args.output}")
    print(f"2. 모델 학습: Colab 노트북에 업로드")


if __name__ == '__main__':
    main()

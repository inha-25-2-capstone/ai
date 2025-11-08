"""
데이터 검증 스크립트

라벨링된 데이터의 품질을 확인하고 문제를 찾아냅니다.

사용법:
    python scripts/validate_data.py --input data/labeled_news.csv
"""

import argparse
import pandas as pd
import numpy as np
from collections import Counter


class DataValidator:
    """데이터 검증 클래스"""

    def __init__(self, filepath):
        """
        Args:
            filepath: CSV 또는 JSON 파일 경로
        """
        print(f"📂 데이터 로딩 중: {filepath}")

        if filepath.endswith('.csv'):
            self.df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            self.df = pd.read_json(filepath)
        else:
            raise ValueError("CSV 또는 JSON 파일만 지원됩니다.")

        print(f"✅ {len(self.df)}개 샘플 로드 완료\n")

    def validate_all(self):
        """전체 검증 실행"""
        print("=" * 60)
        print("📊 데이터 검증 시작")
        print("=" * 60)

        issues = []

        # 1. 기본 정보
        self._check_basic_info()

        # 2. 필수 컬럼 확인
        issues.extend(self._check_required_columns())

        # 3. 레이블 검증
        issues.extend(self._check_labels())

        # 4. 텍스트 품질 검증
        issues.extend(self._check_text_quality())

        # 5. 클래스 균형 확인
        issues.extend(self._check_class_balance())

        # 6. 중복 확인
        issues.extend(self._check_duplicates())

        # 최종 리포트
        print("\n" + "=" * 60)
        if not issues:
            print("✅ 모든 검증 통과! 데이터가 학습에 적합합니다.")
        else:
            print(f"⚠️  {len(issues)}개의 문제가 발견되었습니다:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        print("=" * 60)

        return len(issues) == 0

    def _check_basic_info(self):
        """기본 정보 출력"""
        print("\n📋 기본 정보")
        print(f"   전체 샘플 수: {len(self.df)}개")
        print(f"   컬럼: {', '.join(self.df.columns)}")
        print(f"   메모리 사용: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    def _check_required_columns(self):
        """필수 컬럼 확인"""
        print("\n🔍 필수 컬럼 확인")
        required = ['text', 'label']
        issues = []

        for col in required:
            if col not in self.df.columns:
                issue = f"필수 컬럼 누락: {col}"
                print(f"   ❌ {issue}")
                issues.append(issue)
            else:
                print(f"   ✅ {col}")

        return issues

    def _check_labels(self):
        """레이블 검증"""
        print("\n🏷️  레이블 검증")
        issues = []

        if 'label' not in self.df.columns:
            return [{"필수 컬럼 'label'이 없습니다."}]

        # 결측치 확인
        null_count = self.df['label'].isnull().sum()
        if null_count > 0:
            issue = f"레이블 결측치 {null_count}개 발견"
            print(f"   ⚠️  {issue}")
            issues.append(issue)

        # 유효한 값 확인 (0, 1, 2만 허용)
        valid_labels = {0, 1, 2}
        invalid_labels = set(self.df['label'].dropna().unique()) - valid_labels

        if invalid_labels:
            issue = f"유효하지 않은 레이블 값: {invalid_labels} (0, 1, 2만 허용)"
            print(f"   ❌ {issue}")
            issues.append(issue)
        else:
            print(f"   ✅ 모든 레이블 값이 유효합니다 (0, 1, 2)")

        # 레이블 분포
        label_names = {0: '옹호', 1: '중립', 2: '비판'}
        print(f"\n   레이블 분포:")
        for label in [0, 1, 2]:
            count = (self.df['label'] == label).sum()
            percentage = count / len(self.df) * 100 if len(self.df) > 0 else 0
            label_name = label_names[label]
            print(f"      {label} ({label_name}): {count}개 ({percentage:.1f}%)")

        return issues

    def _check_text_quality(self):
        """텍스트 품질 검증"""
        print("\n📝 텍스트 품질 검증")
        issues = []

        if 'text' not in self.df.columns:
            return ["필수 컬럼 'text'가 없습니다."]

        # 결측치
        null_count = self.df['text'].isnull().sum()
        if null_count > 0:
            issue = f"텍스트 결측치 {null_count}개"
            print(f"   ⚠️  {issue}")
            issues.append(issue)

        # 빈 문자열
        empty_count = (self.df['text'].str.strip() == '').sum()
        if empty_count > 0:
            issue = f"빈 텍스트 {empty_count}개"
            print(f"   ⚠️  {issue}")
            issues.append(issue)

        # 너무 짧은 텍스트 (100자 미만)
        self.df['text_length'] = self.df['text'].fillna('').str.len()
        short_count = (self.df['text_length'] < 100).sum()
        if short_count > 0:
            issue = f"너무 짧은 텍스트 {short_count}개 (100자 미만)"
            print(f"   ⚠️  {issue}")
            issues.append(issue)

        # 텍스트 길이 통계
        print(f"\n   텍스트 길이 통계:")
        print(f"      평균: {self.df['text_length'].mean():.0f}자")
        print(f"      중간값: {self.df['text_length'].median():.0f}자")
        print(f"      최소: {self.df['text_length'].min():.0f}자")
        print(f"      최대: {self.df['text_length'].max():.0f}자")

        return issues

    def _check_class_balance(self):
        """클래스 균형 확인"""
        print("\n⚖️  클래스 균형 확인")
        issues = []

        if 'label' not in self.df.columns:
            return []

        label_counts = self.df['label'].value_counts()

        if len(label_counts) == 0:
            return ["레이블이 없습니다."]

        max_count = label_counts.max()
        min_count = label_counts.min()
        ratio = max_count / min_count if min_count > 0 else float('inf')

        print(f"   최대/최소 비율: {ratio:.2f}:1")

        if ratio > 3.0:
            issue = f"클래스 불균형 심각 (비율: {ratio:.2f}:1, 권장: 3:1 이하)"
            print(f"   ⚠️  {issue}")
            issues.append(issue)
        elif ratio > 2.0:
            print(f"   ⚠️  클래스 불균형 약간 있음 (비율: {ratio:.2f}:1)")
        else:
            print(f"   ✅ 클래스 균형이 양호합니다")

        # 최소 샘플 수 확인
        min_required = 100
        if min_count < min_required:
            issue = f"일부 클래스의 샘플 수가 부족 (최소: {min_count}개, 권장: {min_required}개 이상)"
            print(f"   ⚠️  {issue}")
            issues.append(issue)

        return issues

    def _check_duplicates(self):
        """중복 확인"""
        print("\n🔄 중복 확인")
        issues = []

        if 'text' not in self.df.columns:
            return []

        # 완전 중복
        duplicate_count = self.df.duplicated(subset=['text']).sum()
        if duplicate_count > 0:
            issue = f"완전 중복 텍스트 {duplicate_count}개"
            print(f"   ⚠️  {issue}")
            issues.append(issue)
        else:
            print(f"   ✅ 중복 없음")

        return issues

    def get_recommendations(self):
        """개선 권장사항 출력"""
        print("\n" + "=" * 60)
        print("💡 개선 권장사항")
        print("=" * 60)

        if 'label' in self.df.columns:
            label_counts = self.df['label'].value_counts()
            total = len(self.df)

            if total < 300:
                print(f"\n1. 데이터 양 증가 필요")
                print(f"   현재: {total}개")
                print(f"   권장: 300개 이상 (클래스당 100개)")
                print(f"   이상적: 1,000개 이상 (클래스당 300개)")

            for label in [0, 1, 2]:
                count = label_counts.get(label, 0)
                label_name = ['옹호', '중립', '비판'][label]
                if count < 100:
                    print(f"\n2. '{label_name}' 클래스 데이터 추가 필요")
                    print(f"   현재: {count}개")
                    print(f"   추가 필요: {100 - count}개 이상")

        print("\n3. 데이터 품질 개선")
        print(f"   - 너무 짧은 텍스트 제거 또는 보완")
        print(f"   - 중복 데이터 제거")
        print(f"   - 애매한 레이블 재검토")

        print("\n4. 다음 단계")
        print(f"   - 문제가 해결되면 Colab 노트북에서 학습 시작")
        print(f"   - 학습 후 성능 평가 (목표: 70% 이상)")


def main():
    parser = argparse.ArgumentParser(description='데이터 검증')
    parser.add_argument('--input', type=str, required=True, help='검증할 데이터 파일 경로')

    args = parser.parse_args()

    try:
        validator = DataValidator(args.input)
        is_valid = validator.validate_all()
        validator.get_recommendations()

        # 종료 코드
        exit(0 if is_valid else 1)

    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {args.input}")
        exit(1)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        exit(1)


if __name__ == '__main__':
    main()

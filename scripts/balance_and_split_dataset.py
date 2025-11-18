#!/usr/bin/env python3
"""
클래스 균형 조정 후 train/val/test 분리
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def balance_classes(df, method='undersample', target_size=None, random_state=42):
    """
    클래스 균형 조정

    Args:
        df: 원본 DataFrame
        method: 'undersample' (다운샘플링) 또는 'target' (목표 크기 지정)
        target_size: 각 클래스의 목표 크기 (method='target'일 때)
        random_state: 랜덤 시드
    """
    print(f"\n{'='*80}")
    print("클래스 균형 조정")
    print(f"{'='*80}\n")

    # 원본 라벨 분포
    print("원본 라벨 분포:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = ['옹호', '중립', '비판'][label]
        print(f"  {label} ({label_name}): {count}개 ({count/len(df)*100:.1f}%)")

    # 목표 크기 결정
    if method == 'undersample':
        # 가장 적은 클래스 크기로 맞추기
        target_size = label_counts.min()
        print(f"\n다운샘플링: 각 클래스를 {target_size}개로 맞춤")
    elif target_size is not None:
        print(f"\n목표 크기: 각 클래스를 {target_size}개로 맞춤")
    else:
        print("ERROR: method='target'일 때는 target_size가 필요합니다.")
        return None

    # 각 클래스별로 샘플링
    balanced_dfs = []

    for label in sorted(df['label'].unique()):
        label_df = df[df['label'] == label]
        current_size = len(label_df)

        if current_size > target_size:
            # 다운샘플링
            sampled_df = label_df.sample(n=target_size, random_state=random_state)
            print(f"  라벨 {label}: {current_size}개 → {target_size}개 (다운샘플링)")
        elif current_size < target_size:
            # 오버샘플링 (복원 추출)
            sampled_df = label_df.sample(n=target_size, replace=True, random_state=random_state)
            print(f"  라벨 {label}: {current_size}개 → {target_size}개 (오버샘플링)")
        else:
            sampled_df = label_df
            print(f"  라벨 {label}: {current_size}개 (유지)")

        balanced_dfs.append(sampled_df)

    # 합치기
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    # 셔플
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"\n균형 조정 후:")
    print(f"  전체 데이터: {len(balanced_df)}개")

    balanced_label_counts = balanced_df['label'].value_counts().sort_index()
    for label, count in balanced_label_counts.items():
        label_name = ['옹호', '중립', '비판'][label]
        print(f"  {label} ({label_name}): {count}개 ({count/len(balanced_df)*100:.1f}%)")

    return balanced_df


def split_dataset(
    input_csv,
    output_dir=None,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    balance_method='undersample',
    target_size=None,
    random_state=42
):
    """
    클래스 균형 조정 후 데이터셋을 train/val/test로 분리
    """
    print(f"\n{'='*80}")
    print("균형 조정 + 데이터셋 분리 (train/val/test)")
    print(f"{'='*80}\n")

    # 비율 검증
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"ERROR: 비율의 합이 1.0이 아닙니다: {total_ratio}")
        return

    # CSV 읽기
    input_path = Path(input_csv)
    print(f"입력 파일: {input_path}")

    df = pd.read_csv(input_path)
    print(f"전체 데이터: {len(df)}개")

    # 클래스 균형 조정
    balanced_df = balance_classes(df, method=balance_method, target_size=target_size, random_state=random_state)

    if balanced_df is None:
        return

    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = input_path.parent / "balanced_splits"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1단계: train + (val+test) 분리
    train_df, temp_df = train_test_split(
        balanced_df,
        test_size=(val_ratio + test_ratio),
        stratify=balanced_df['label'],
        random_state=random_state
    )

    # 2단계: val + test 분리
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_df['label'],
        random_state=random_state
    )

    print(f"\n{'='*80}")
    print("분리 결과:")
    print(f"{'='*80}")

    # Train 세트
    print(f"\nTrain 세트: {len(train_df)}개 ({len(train_df)/len(balanced_df)*100:.1f}%)")
    train_label_counts = train_df['label'].value_counts().sort_index()
    for label, count in train_label_counts.items():
        label_name = ['옹호', '중립', '비판'][label]
        print(f"  {label} ({label_name}): {count}개 ({count/len(train_df)*100:.1f}%)")

    # Validation 세트
    print(f"\nValidation 세트: {len(val_df)}개 ({len(val_df)/len(balanced_df)*100:.1f}%)")
    val_label_counts = val_df['label'].value_counts().sort_index()
    for label, count in val_label_counts.items():
        label_name = ['옹호', '중립', '비판'][label]
        print(f"  {label} ({label_name}): {count}개 ({count/len(val_df)*100:.1f}%)")

    # Test 세트
    print(f"\nTest 세트: {len(test_df)}개 ({len(test_df)/len(balanced_df)*100:.1f}%)")
    test_label_counts = test_df['label'].value_counts().sort_index()
    for label, count in test_label_counts.items():
        label_name = ['옹호', '중립', '비판'][label]
        print(f"  {label} ({label_name}): {count}개 ({count/len(test_df)*100:.1f}%)")

    # 파일 저장
    print(f"\n{'='*80}")
    print("파일 저장 중...")
    print(f"{'='*80}\n")

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    print(f"Train: {train_path} ({train_path.stat().st_size / (1024*1024):.2f} MB)")

    val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
    print(f"Val:   {val_path} ({val_path.stat().st_size / (1024*1024):.2f} MB)")

    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
    print(f"Test:  {test_path} ({test_path.stat().st_size / (1024*1024):.2f} MB)")

    print(f"\n{'='*80}")
    print("완료!")
    print(f"{'='*80}\n")

    return train_path, val_path, test_path


def main():
    parser = argparse.ArgumentParser(
        description="클래스 균형 조정 후 데이터셋 분리 (train/val/test)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="data/batch_results/politics_labeled_dataset.csv",
        help="입력 CSV 파일 경로"
    )
    parser.add_argument(
        "--output-dir",
        help="출력 디렉토리 (기본: 입력 파일과 같은 디렉토리의 balanced_splits 폴더)"
    )
    parser.add_argument(
        "--balance-method",
        choices=['undersample', 'target'],
        default='undersample',
        help="균형 조정 방법: undersample(가장 적은 클래스 기준) 또는 target(목표 크기 지정)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        help="각 클래스의 목표 크기 (balance-method=target일 때 필수)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train 비율 (기본: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation 비율 (기본: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test 비율 (기본: 0.1)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본: 42)"
    )

    args = parser.parse_args()

    split_dataset(
        input_csv=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        balance_method=args.balance_method,
        target_size=args.target_size,
        random_state=args.random_seed
    )


if __name__ == "__main__":
    main()

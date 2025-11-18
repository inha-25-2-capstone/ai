#!/usr/bin/env python3
"""
파인튜닝용 데이터셋 분리 (train/val/test)
라벨 비율을 동일하게 유지하는 stratified split
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset(
    input_csv,
    output_dir=None,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_state=42
):
    """
    데이터셋을 train/val/test로 분리

    Args:
        input_csv: 입력 CSV 파일 경로
        output_dir: 출력 디렉토리
        train_ratio: train 비율 (기본 0.8 = 80%)
        val_ratio: validation 비율 (기본 0.1 = 10%)
        test_ratio: test 비율 (기본 0.1 = 10%)
        random_state: 랜덤 시드 (재현성)
    """
    print(f"\n{'='*80}")
    print("데이터셋 분리 (train/val/test)")
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

    # 라벨 분포 확인
    print(f"\n전체 라벨 분포:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = ['옹호', '중립', '비판'][label]
        print(f"  {label} ({label_name}): {count}개 ({count/len(df)*100:.1f}%)")

    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = input_path.parent / "splits"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1단계: train + (val+test) 분리
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['label'],
        random_state=random_state
    )

    # 2단계: val + test 분리
    # temp_df에서 val과 test의 비율 계산
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
    print(f"\nTrain 세트: {len(train_df)}개 ({len(train_df)/len(df)*100:.1f}%)")
    train_label_counts = train_df['label'].value_counts().sort_index()
    for label, count in train_label_counts.items():
        label_name = ['옹호', '중립', '비판'][label]
        print(f"  {label} ({label_name}): {count}개 ({count/len(train_df)*100:.1f}%)")

    # Validation 세트
    print(f"\nValidation 세트: {len(val_df)}개 ({len(val_df)/len(df)*100:.1f}%)")
    val_label_counts = val_df['label'].value_counts().sort_index()
    for label, count in val_label_counts.items():
        label_name = ['옹호', '중립', '비판'][label]
        print(f"  {label} ({label_name}): {count}개 ({count/len(val_df)*100:.1f}%)")

    # Test 세트
    print(f"\nTest 세트: {len(test_df)}개 ({len(test_df)/len(df)*100:.1f}%)")
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

    # 검증: 비율 확인
    print("비율 검증:")
    print(f"  Train: {len(train_df)/len(df)*100:.2f}% (목표: {train_ratio*100:.1f}%)")
    print(f"  Val:   {len(val_df)/len(df)*100:.2f}% (목표: {val_ratio*100:.1f}%)")
    print(f"  Test:  {len(test_df)/len(df)*100:.2f}% (목표: {test_ratio*100:.1f}%)")

    return train_path, val_path, test_path


def main():
    parser = argparse.ArgumentParser(
        description="파인튜닝용 데이터셋 분리 (train/val/test)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="data/batch_results/politics_labeled_dataset.csv",
        help="입력 CSV 파일 경로"
    )
    parser.add_argument(
        "--output-dir",
        help="출력 디렉토리 (기본: 입력 파일과 같은 디렉토리의 splits 폴더)"
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
        random_state=args.random_seed
    )


if __name__ == "__main__":
    main()

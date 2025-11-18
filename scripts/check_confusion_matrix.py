#!/usr/bin/env python3
"""
테스트 데이터로 혼동 행렬 확인
"""
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import sys
sys.path.append('.')

from scripts.predict_stance import StancePredictor

# 모델 로드
predictor = StancePredictor(
    model_path='saved_models/stance_classifier_20251117_125252.pth',
    model_name='skt/kobert-base-v1',
    max_length=512
)

# 테스트 데이터 로드
test_df = pd.read_csv('data/batch_results/balanced_splits/test.csv')
print(f"테스트 데이터: {len(test_df)}개")
print(f"라벨 분포:")
print(test_df['label'].value_counts().sort_index())

# 예측
texts = test_df['content'].tolist()
true_labels = test_df['label'].tolist()

print("\n예측 중...")
results = predictor.predict_batch(texts, batch_size=16, show_progress=True)
predicted_labels = [r['predicted_label'] for r in results]

# 혼동 행렬
cm = confusion_matrix(true_labels, predicted_labels)
print("\n=== 혼동 행렬 ===")
print("              예측")
print("실제    옹호  중립  비판")
for i, label_name in enumerate(['옹호', '중립', '비판']):
    print(f"{label_name:4s}  {cm[i][0]:4d} {cm[i][1]:4d} {cm[i][2]:4d}")

# 분류 보고서
print("\n=== 분류 보고서 ===")
target_names = ['옹호 (Support)', '중립 (Neutral)', '비판 (Oppose)']
print(classification_report(true_labels, predicted_labels, target_names=target_names, digits=4))

# 각 라벨별 예측 분포
print("\n=== 예측 분포 ===")
pred_df = pd.DataFrame({'true': true_labels, 'pred': predicted_labels})
for label in [0, 1, 2]:
    label_name = ['옹호', '중립', '비판'][label]
    subset = pred_df[pred_df['true'] == label]
    pred_counts = subset['pred'].value_counts().sort_index()
    print(f"\n실제 {label_name} ({len(subset)}개):")
    for pred_label in [0, 1, 2]:
        pred_label_name = ['옹호', '중립', '비판'][pred_label]
        count = pred_counts.get(pred_label, 0)
        print(f"  → {pred_label_name}로 예측: {count}개 ({count/len(subset)*100:.1f}%)")

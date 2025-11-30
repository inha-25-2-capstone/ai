# -*- coding: utf-8 -*-
"""
KoBERT 기반 뉴스 스탠스 분류 모델 학습 스크립트
- 옹호(Support): 0
- 중립(Neutral): 1
- 비판(Oppose): 2
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import BertModel, AutoTokenizer

import pandas as pd
import numpy as np
import random
import argparse

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import json
import os

# wandb
import wandb


def set_seed(seed=42):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StanceDataset(Dataset):
    """스탠스 분류용 데이터셋"""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class StanceClassifier(nn.Module):
    """KoBERT 기반 스탠스 분류 모델"""
    def __init__(self, n_classes=3, dropout=0.3):
        super(StanceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('skt/kobert-base-v1')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)


def train_epoch(model, data_loader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


def eval_model(model, data_loader, criterion, device):
    """모델 평가"""
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


def get_predictions(model, data_loader, device):
    """예측값 추출"""
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Getting predictions'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(true_labels)


def plot_training_history(history, save_dir):
    """학습 곡선 저장"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(history['train_acc'], label='Train Accuracy')
    axes[0].plot(history['val_acc'], label='Val Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history['train_loss'], label='Train Loss')
    axes[1].plot(history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()


def plot_confusion_matrix(true_labels, predictions, save_dir):
    """Confusion Matrix 저장"""
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Support', 'Neutral', 'Oppose'],
                yticklabels=['Support', 'Neutral', 'Oppose'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # 정규화된 Confusion Matrix
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Support', 'Neutral', 'Oppose'],
                yticklabels=['Support', 'Neutral', 'Oppose'])
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.png'), dpi=150)
    plt.close()


def main(args):
    # 시드 설정
    set_seed(args.seed)

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # wandb 초기화
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"kobert-stance-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            'model': 'skt/kobert-base-v1',
            'max_length': args.max_length,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'dropout': args.dropout,
            'seed': args.seed,
            'patience': args.patience
        }
    )

    # 데이터 로드
    print(f"\n[INFO] Loading data from: {args.data_path}")

    if args.data_path.endswith('.csv'):
        df = pd.read_csv(args.data_path)
    elif args.data_path.endswith('.json'):
        df = pd.read_json(args.data_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .json")

    # 컬럼 확인 및 매핑
    if 'text' not in df.columns and 'content' in df.columns:
        df['text'] = df['content']

    print(f"[INFO] Total samples: {len(df)}")
    print(f"[INFO] Columns: {df.columns.tolist()}")

    # 라벨 분포
    print("\n[INFO] Label distribution:")
    label_names = ['Support (0)', 'Neutral (1)', 'Oppose (2)']
    for label_id, count in df['label'].value_counts().sort_index().items():
        print(f"  {label_names[label_id]}: {count} ({count/len(df)*100:.1f}%)")

    # 데이터 분할
    train_df, test_df = train_test_split(
        df, test_size=0.1, random_state=args.seed, stratify=df['label']
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.111, random_state=args.seed, stratify=train_df['label']
    )  # 0.111 * 0.9 = 0.1 for val

    print(f"\n[INFO] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 토크나이저 로드
    print("\n[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')

    # 데이터셋 생성
    train_texts = train_df['text'].values
    train_labels = train_df['label'].values
    val_texts = val_df['text'].values
    val_labels = val_df['label'].values
    test_texts = test_df['text'].values
    test_labels = test_df['label'].values

    train_dataset = StanceDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = StanceDataset(val_texts, val_labels, tokenizer, args.max_length)
    test_dataset = StanceDataset(test_texts, test_labels, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # 모델 생성
    print("\n[INFO] Creating model...")
    model = StanceClassifier(n_classes=3, dropout=args.dropout)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")

    # 클래스 가중치 계산
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"[INFO] Class weights: {class_weights.tolist()}")

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # 저장 디렉토리
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    # 학습 기록
    history = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': []
    }

    best_val_acc = 0
    best_val_loss = float('inf')
    best_model_path = None
    early_stop_counter = 0

    # 학습 시작
    print("\n" + "="*60)
    print("TRAINING START")
    print(f"Early Stopping Patience: {args.patience}")
    print("="*60 + "\n")

    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        print('-' * 50)

        train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_loss = eval_model(model, val_loader, criterion, device)

        history['train_acc'].append(train_acc.item())
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc.item())
        history['val_loss'].append(val_loss)

        # wandb 로깅
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc.item(),
            'val_loss': val_loss,
            'val_acc': val_acc.item(),
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%')

        # Best model 저장 (val_loss 기준)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'[SAVE] Best model saved! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%)')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f'[INFO] No improvement. Early stop counter: {early_stop_counter}/{args.patience}')

        scheduler.step(val_loss)

        # Early Stopping 체크
        if early_stop_counter >= args.patience:
            print(f'\n[EARLY STOPPING] No improvement for {args.patience} epochs. Stopping training.')
            break

        print()

    print("="*60)
    print("TRAINING COMPLETE")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Total Epochs: {epoch + 1}")
    print("="*60)

    # Best 모델 로드 후 테스트
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))

    test_acc, test_loss = eval_model(model, test_loader, criterion, device)

    print(f"\n[TEST RESULT]")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # wandb 테스트 결과 로깅
    wandb.log({
        'test_loss': test_loss,
        'test_acc': test_acc.item()
    })

    # 예측값 추출 및 분류 보고서
    predictions, true_labels = get_predictions(model, test_loader, device)

    target_names = ['Support', 'Neutral', 'Oppose']
    report = classification_report(true_labels, predictions, target_names=target_names, digits=4, zero_division=0)
    print("\n[CLASSIFICATION REPORT]")
    print(report)

    # wandb에 분류 보고서 로깅
    report_dict = classification_report(true_labels, predictions, target_names=target_names, output_dict=True, zero_division=0)
    wandb.log({
        'test_precision_macro': report_dict['macro avg']['precision'],
        'test_recall_macro': report_dict['macro avg']['recall'],
        'test_f1_macro': report_dict['macro avg']['f1-score']
    })

    # 시각화 저장
    plot_training_history(history, save_dir)
    plot_confusion_matrix(true_labels, predictions, save_dir)
    print(f"\n[INFO] Plots saved to {save_dir}")

    # 메타데이터 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metadata = {
        'timestamp': timestamp,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'best_val_accuracy': float(best_val_acc),
        'hyperparameters': {
            'max_length': args.max_length,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'dropout': args.dropout,
            'seed': args.seed,
            'patience': args.patience
        },
        'data_info': {
            'total_samples': len(df),
            'train_samples': len(train_texts),
            'val_samples': len(val_texts),
            'test_samples': len(test_texts)
        },
        'model_name': 'skt/kobert-base-v1'
    }

    metadata_path = os.path.join(save_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Metadata saved to {metadata_path}")

    # wandb 종료
    wandb.finish()

    print("\n[DONE] Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KoBERT Stance Classification Training')

    # 데이터 관련
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset (CSV or JSON)')
    parser.add_argument('--output_dir', type=str, default='outputs/stance_model',
                        help='Output directory for model and logs')

    # 하이퍼파라미터
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')

    # wandb 관련
    parser.add_argument('--wandb_project', type=str, default='stance-classification',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')

    args = parser.parse_args()
    main(args)

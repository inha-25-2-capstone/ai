"""
KoBERT 기반 뉴스 스탠스 분류 모델 학습 스크립트 (개선 버전 v2)

개선 사항:
1. Class Weighted Loss
2. Focal Loss 옵션
3. Early Stopping (강화)
4. Weight Decay (L2 regularization)
5. Dropout 증가 (0.3 → 0.4)
6. Gradient Clipping

Usage:
    python scripts/train_kobert_improved.py --data_dir data --output_dir models/kobert_stance
"""

import argparse
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AutoTokenizer

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm.auto import tqdm

# 랜덤 시드 고정
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class EarlyStopping:
    """
    Early Stopping: 검증 손실이 개선되지 않으면 학습 중단
    """
    def __init__(self, patience=3, min_delta=0.001, verbose=True):
        """
        Args:
            patience: 개선이 없어도 기다릴 epoch 수
            min_delta: 개선으로 간주할 최소 변화량
            verbose: 로그 출력 여부
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        """검증 손실을 체크하고 early stop 여부 결정"""
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)

        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'⚠️  EarlyStopping counter: {self.counter}/{self.patience}')
                print(f'   현재 Val Loss: {val_loss:.4f}, 최고 Val Loss: {self.best_loss:.4f}')

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'\n🛑 Early Stopping! {self.patience} epoch 동안 개선 없음')

        else:
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """최고 성능 모델 저장"""
        if self.verbose:
            if self.best_loss is not None:
                print(f'✅ Val Loss 개선: {self.best_loss:.4f} → {val_loss:.4f}')
            else:
                print(f'✅ 최초 모델 저장 (Val Loss: {val_loss:.4f})')

        self.best_loss = val_loss
        # 모델 state dict 저장 (메모리에)
        self.best_model_state = {
            key: value.cpu().clone() for key, value in model.state_dict().items()
        }

    def load_best_model(self, model):
        """최고 성능 모델 로드"""
        if self.best_model_state is not None:
            model.load_state_dict(
                {key: value.to(next(model.parameters()).device)
                 for key, value in self.best_model_state.items()}
            )
            if self.verbose:
                print(f'✅ 최고 성능 모델 로드 완료 (Val Loss: {self.best_loss:.4f})')
        return model


class FocalLoss(nn.Module):
    """
    Focal Loss - 어려운 샘플에 집중하도록 하는 손실 함수
    클래스 불균형 문제에 효과적
    """
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 클래스별 가중치
        self.gamma = gamma  # focusing parameter

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class StanceDataset(Dataset):
    """스탠스 분류 데이터셋"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
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

    def __init__(self, n_classes=3, dropout=0.4, model_name='skt/kobert-base-v1'):
        super(StanceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # [CLS] 토큰의 출력 사용
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


def calculate_class_weights(labels, device):
    """
    클래스별 가중치 계산
    적은 클래스에 더 높은 가중치 부여
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    # 역빈도 가중치 계산
    weights = total / (len(unique) * counts)

    # 정규화
    weights = weights / weights.sum() * len(unique)

    class_weights = torch.FloatTensor(weights).to(device)

    print("\n클래스별 가중치:")
    label_names = ['옹호', '중립', '비판']
    for i, (count, weight) in enumerate(zip(counts, weights)):
        print(f"  {label_names[i]}: {count}개 → 가중치 {weight:.4f}")

    return class_weights


def train_epoch(model, data_loader, criterion, optimizer, device, epoch_num, max_grad_norm=1.0):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # 클래스별 예측 카운트
    class_predictions = {0: 0, 1: 0, 2: 0}

    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch_num} Training')

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        # 그래디언트 클리핑 (Exploding gradient 방지)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 예측 클래스 카운트
        for pred in predicted.cpu().numpy():
            class_predictions[pred] += 1

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    # 클래스별 예측 분포 출력
    print(f"\n  예측 분포: 옹호={class_predictions[0]}, 중립={class_predictions[1]}, 비판={class_predictions[2]}")

    return avg_loss, accuracy


def eval_model(model, data_loader, criterion, device, desc='Evaluating'):
    """모델 평가"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def get_predictions(model, data_loader, device):
    """예측값 얻기"""
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


def main(args):
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"사용 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"{'='*60}\n")

    # 데이터 로드
    print("데이터 로딩 중...")
    train_df = pd.read_csv(os.path.join(args.data_dir, 'train_dataset.csv'))
    val_df = pd.read_csv(os.path.join(args.data_dir, 'val_dataset.csv'))
    test_df = pd.read_csv(os.path.join(args.data_dir, 'test_dataset.csv'))

    print(f"\n학습 데이터: {len(train_df)}개")
    print(f"검증 데이터: {len(val_df)}개")
    print(f"테스트 데이터: {len(test_df)}개")

    # 라벨 분포 출력
    print(f"\n학습 데이터 라벨 분포:")
    for label_id, count in train_df['label'].value_counts().sort_index().items():
        label_name = ['옹호', '중립', '비판'][label_id]
        print(f"  {label_id} ({label_name}): {count}개 ({count/len(train_df)*100:.1f}%)")

    # 클래스 가중치 계산
    class_weights = calculate_class_weights(train_df['label'].values, device)

    # 토크나이저 로드
    print(f"\n토크나이저 로딩 중... ({args.model_name})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    except:
        print("대안 토크나이저 사용: monologg/kobert")
        tokenizer = AutoTokenizer.from_pretrained('monologg/kobert')
    print(f"Vocab 크기: {len(tokenizer)}")

    # Dataset 생성
    train_dataset = StanceDataset(
        train_df['text'].values,
        train_df['label'].values,
        tokenizer,
        args.max_length
    )
    val_dataset = StanceDataset(
        val_df['text'].values,
        val_df['label'].values,
        tokenizer,
        args.max_length
    )
    test_dataset = StanceDataset(
        test_df['text'].values,
        test_df['label'].values,
        tokenizer,
        args.max_length
    )

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 모델 초기화
    print(f"\n모델 로딩 중...")
    model = StanceClassifier(
        n_classes=3,
        dropout=args.dropout,
        model_name=args.model_name
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"전체 파라미터: {total_params:,}")
    print(f"학습 가능한 파라미터: {trainable_params:,}")

    # 손실 함수 선택
    if args.use_focal_loss:
        print("\n손실 함수: Focal Loss (gamma={})".format(args.focal_gamma))
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    else:
        print("\n손실 함수: Weighted Cross Entropy")
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 옵티마이저 (Weight Decay 명시적 설정!)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,  # L2 regularization
        eps=1e-8
    )

    # 학습률 스케줄러 (validation loss 기준으로 변경)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',        # validation loss를 최소화
        factor=0.5,        # LR을 절반으로
        patience=2,        # 2 epoch 동안 개선 없으면 LR 감소
        verbose=True,
        min_lr=1e-7        # 최소 learning rate
    )

    # Early Stopping 초기화
    early_stopping = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=0.001,
        verbose=True
    )

    # 학습 기록
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []  # LR 변화 추적
    }

    # 학습 루프
    print(f"\n{'='*60}")
    print("학습 시작 (Early Stopping 활성화)")
    print(f"{'='*60}")
    print(f"최대 Epoch: {args.epochs}")
    print(f"Early Stop Patience: {args.early_stop_patience}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Dropout: {args.dropout}")
    print(f"Max Grad Norm: {args.max_grad_norm}")
    print("="*60 + "\n")

    best_val_f1 = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # 현재 Learning Rate 출력
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📊 Current Learning Rate: {current_lr:.2e}")

        # 학습
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1,
            max_grad_norm=args.max_grad_norm
        )
        print(f"\n📊 Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")

        # 검증
        val_loss, val_acc = eval_model(model, val_loader, criterion, device, 'Validating')

        # 검증 F1 계산
        val_preds, val_true = get_predictions(model, val_loader, device)
        from sklearn.metrics import f1_score
        val_f1_macro = f1_score(val_true, val_preds, average='macro')

        print(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val F1 (macro): {val_f1_macro:.4f}")

        # 학습률 스케줄러 업데이트 (Val Loss 기준으로 변경)
        scheduler.step(val_loss)

        # 기록 저장
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)

        # 최고 F1 성능 추적
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_epoch = epoch + 1
            print(f"✨ 새로운 최고 검증 F1: {best_val_f1:.4f}")

        # Early Stopping 체크 (Val Loss 기준)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print(f"\n{'='*60}")
            print(f"🛑 Early Stopping at Epoch {epoch + 1}")
            print(f"{'='*60}")
            break

    # 최고 성능 모델 로드
    print(f"\n{'='*60}")
    print("최고 성능 모델 로드")
    print(f"{'='*60}")
    model = early_stopping.load_best_model(model)
    print(f"✅ 최고 검증 F1: {best_val_f1:.4f} (Epoch {best_epoch})")

    print(f"\n{'='*60}")
    print("학습 완료!")
    print(f"{'='*60}")
    print(f"총 학습 Epoch: {epoch + 1}/{args.epochs}")
    print(f"최고 검증 F1: {best_val_f1:.4f} (Epoch {best_epoch})")
    print(f"최종 Val Loss: {early_stopping.best_loss:.4f}")

    # 테스트 평가
    print(f"\n{'='*60}")
    print("테스트 평가")
    print(f"{'='*60}\n")

    test_loss, test_acc = eval_model(model, test_loader, criterion, device, 'Testing')
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # 상세 분류 보고서
    predictions, true_labels = get_predictions(model, test_loader, device)
    target_names = ['옹호 (Support)', '중립 (Neutral)', '비판 (Oppose)']
    print("\n분류 보고서:")
    print(classification_report(
        true_labels, predictions,
        target_names=target_names,
        digits=4,
        zero_division=0
    ))

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    print("\nConfusion Matrix:")
    print("              Predicted")
    print("              Support  Neutral  Oppose")
    print(f"Actual Support   {cm[0][0]:3d}      {cm[0][1]:3d}      {cm[0][2]:3d}")
    print(f"       Neutral   {cm[1][0]:3d}      {cm[1][1]:3d}      {cm[1][2]:3d}")
    print(f"       Oppose    {cm[2][0]:3d}      {cm[2][1]:3d}      {cm[2][2]:3d}")

    # 모델 저장
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'kobert_stance_improved_{timestamp}.pth'
    model_path = os.path.join(args.output_dir, model_filename)

    torch.save(model.state_dict(), model_path)
    print(f"\n모델 저장 완료: {model_path}")

    # 메타데이터 저장
    metadata = {
        'timestamp': timestamp,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'best_val_f1_macro': float(best_val_f1),
        'hyperparameters': {
            'max_length': args.max_length,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'dropout': args.dropout,
            'use_focal_loss': args.use_focal_loss,
            'focal_gamma': args.focal_gamma if args.use_focal_loss else None
        },
        'data_info': {
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df)
        },
        'model_name': args.model_name,
        'class_weights': class_weights.cpu().tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            true_labels, predictions,
            target_names=target_names,
            digits=4,
            zero_division=0,
            output_dict=True
        )
    }

    metadata_filename = f'metadata_improved_{timestamp}.json'
    metadata_path = os.path.join(args.output_dir, metadata_filename)

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"메타데이터 저장 완료: {metadata_path}")

    # 학습 히스토리 저장
    history_filename = f'history_improved_{timestamp}.json'
    history_path = os.path.join(args.output_dir, history_filename)

    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"학습 히스토리 저장 완료: {history_path}")

    print(f"\n{'='*60}")
    print("학습 완료!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KoBERT 스탠스 분류 모델 학습 (개선 버전)')

    # 데이터 관련
    parser.add_argument('--data_dir', type=str, default='data',
                        help='데이터 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, default='models/kobert_stance',
                        help='모델 저장 디렉토리 경로')

    # 모델 관련
    parser.add_argument('--model_name', type=str, default='skt/kobert-base-v1',
                        help='사용할 KoBERT 모델명')
    parser.add_argument('--max_length', type=int, default=512,
                        help='최대 시퀀스 길이')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='드롭아웃 비율 (과적합 방지 강화: 0.4)')

    # 학습 관련
    parser.add_argument('--batch_size', type=int, default=16,
                        help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='학습률')
    parser.add_argument('--epochs', type=int, default=15,
                        help='에폭 수 (최대)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Gradient clipping max norm')

    # Early Stopping 관련
    parser.add_argument('--early_stop_patience', type=int, default=3,
                        help='Early stopping patience (개선 없을 때 기다릴 epoch 수)')
    parser.add_argument('--patience', type=int, default=5,
                        help='(Deprecated) LR scheduler patience')

    # 클래스 불균형 대응
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Focal Loss 사용 (기본값: Weighted CE)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss gamma 값')

    args = parser.parse_args()

    main(args)

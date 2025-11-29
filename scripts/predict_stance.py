"""
스탠스 분석 예측 스크립트

학습된 KoBERT 모델을 사용하여 뉴스 기사의 스탠스를 예측합니다.

Usage:
    # 단일 텍스트 예측
    python scripts/predict_stance.py --text "대통령이 새로운 정책을 발표했다."

    # CSV 파일 배치 예측
    python scripts/predict_stance.py --input_file data/test.csv --output_file results.csv

    # 대화형 모드
    python scripts/predict_stance.py --interactive
"""

import argparse
import os
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer
import pandas as pd
from tqdm import tqdm


class StanceClassifier(nn.Module):
    """KoBERT 기반 스탠스 분류 모델"""

    def __init__(self, n_classes=3, dropout=0.3, model_name='skt/kobert-base-v1'):
        super(StanceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class StancePredictor:
    """스탠스 예측기"""

    def __init__(self, model_path, model_name='skt/kobert-base-v1', max_length=512, device=None):
        """
        Args:
            model_path: 학습된 모델 파일 경로 (.pth)
            model_name: 사용할 KoBERT 모델명
            max_length: 최대 시퀀스 길이
            device: 사용할 디바이스 (None이면 자동 선택)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 라벨 정의
        self.label_names = ['옹호', '중립', '비판']
        self.label_names_en = ['Support', 'Neutral', 'Oppose']

        # 토크나이저 로드
        print(f"토크나이저 로딩 중... ({model_name})")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except:
            print("대안 토크나이저 사용: monologg/kobert")
            self.tokenizer = AutoTokenizer.from_pretrained('monologg/kobert', trust_remote_code=True)

        # 모델 로드
        print(f"모델 로딩 중... ({model_path})")
        self.model = StanceClassifier(n_classes=3, dropout=0.3, model_name=model_name)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[OK] 모델 로드 완료! (Device: {self.device})")

    def predict(self, text, return_probs=False):
        """
        단일 텍스트의 스탠스 예측

        Args:
            text: 예측할 텍스트
            return_probs: True이면 확률값도 반환

        Returns:
            예측 결과 딕셔너리
        """
        # 토크나이징
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 예측
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()

        result = {
            'text': text,
            'predicted_label': predicted_class,
            'predicted_label_name': self.label_names[predicted_class],
            'predicted_label_name_en': self.label_names_en[predicted_class],
            'confidence': confidence
        }

        if return_probs:
            result['probabilities'] = {
                self.label_names[i]: probs[0][i].item()
                for i in range(len(self.label_names))
            }

        return result

    def predict_batch(self, texts, batch_size=16, show_progress=True):
        """
        여러 텍스트의 스탠스 배치 예측

        Args:
            texts: 예측할 텍스트 리스트
            batch_size: 배치 크기
            show_progress: 진행률 표시 여부

        Returns:
            예측 결과 리스트
        """
        results = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc='예측 중')

        for i in iterator:
            batch_texts = texts[i:i+batch_size]

            # 토크나이징
            encodings = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)

            # 예측
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probs, dim=1)
                confidences = torch.max(probs, dim=1).values

            # 결과 저장
            for j, text in enumerate(batch_texts):
                pred_class = predicted_classes[j].item()
                results.append({
                    'text': text,
                    'predicted_label': pred_class,
                    'predicted_label_name': self.label_names[pred_class],
                    'predicted_label_name_en': self.label_names_en[pred_class],
                    'confidence': confidences[j].item(),
                    'prob_support': probs[j][0].item(),
                    'prob_neutral': probs[j][1].item(),
                    'prob_oppose': probs[j][2].item()
                })

        return results

    def predict_csv(self, input_file, output_file, text_column='text', batch_size=16):
        """
        CSV 파일의 텍스트들을 예측하고 결과를 CSV로 저장

        Args:
            input_file: 입력 CSV 파일 경로
            output_file: 출력 CSV 파일 경로
            text_column: 텍스트가 있는 컬럼명
            batch_size: 배치 크기
        """
        print(f"\nCSV 파일 로딩 중: {input_file}")
        df = pd.read_csv(input_file)

        if text_column not in df.columns:
            raise ValueError(f"'{text_column}' 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {df.columns.tolist()}")

        print(f"총 {len(df)}개 텍스트 예측 시작...")

        # 배치 예측
        results = self.predict_batch(df[text_column].tolist(), batch_size=batch_size)

        # 결과를 데이터프레임에 추가
        result_df = df.copy()
        result_df['predicted_label'] = [r['predicted_label'] for r in results]
        result_df['predicted_label_name'] = [r['predicted_label_name'] for r in results]
        result_df['confidence'] = [r['confidence'] for r in results]
        result_df['prob_support'] = [r['prob_support'] for r in results]
        result_df['prob_neutral'] = [r['prob_neutral'] for r in results]
        result_df['prob_oppose'] = [r['prob_oppose'] for r in results]

        # 저장
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n[OK] 예측 완료! 결과 저장: {output_file}")

        # 예측 분포 출력
        print("\n예측 결과 분포:")
        for label_name in self.label_names:
            count = (result_df['predicted_label_name'] == label_name).sum()
            print(f"  {label_name}: {count}개 ({count/len(result_df)*100:.1f}%)")

        return result_df


def interactive_mode(predictor):
    """대화형 예측 모드"""
    print("\n" + "="*60)
    print("스탠스 분석 대화형 모드")
    print("="*60)
    print("텍스트를 입력하면 스탠스를 예측합니다.")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("="*60 + "\n")

    while True:
        text = input("\n텍스트 입력 > ").strip()

        if text.lower() in ['quit', 'exit', '종료']:
            print("종료합니다.")
            break

        if not text:
            print("텍스트를 입력해주세요.")
            continue

        # 예측
        result = predictor.predict(text, return_probs=True)

        # 결과 출력
        print(f"\n{'-'*60}")
        print(f"[TEXT] {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"{'-'*60}")
        print(f"[PREDICTION] {result['predicted_label_name']} ({result['predicted_label_name_en']})")
        print(f"[CONFIDENCE] {result['confidence']*100:.1f}%")
        print(f"\n[PROBABILITIES]")
        for label, prob in result['probabilities'].items():
            bar = "#" * int(prob * 50)
            print(f"  {label:4s}: {bar} {prob*100:.1f}%")
        print(f"{'-'*60}")


def main():
    parser = argparse.ArgumentParser(description='KoBERT 스탠스 분석 예측')

    # 모델 관련
    parser.add_argument('--model_path', type=str,
                        default='saved_models/stance_classifier_20251117_125252.pth',
                        help='학습된 모델 파일 경로')
    parser.add_argument('--model_name', type=str, default='skt/kobert-base-v1',
                        help='사용할 KoBERT 모델명')
    parser.add_argument('--max_length', type=int, default=512,
                        help='최대 시퀀스 길이')

    # 예측 모드
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--text', type=str, help='예측할 단일 텍스트')
    group.add_argument('--input_file', type=str, help='입력 CSV 파일 경로')
    group.add_argument('--interactive', action='store_true', help='대화형 모드')

    # CSV 관련
    parser.add_argument('--output_file', type=str, help='출력 CSV 파일 경로')
    parser.add_argument('--text_column', type=str, default='text',
                        help='텍스트가 있는 컬럼명')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='배치 크기')

    args = parser.parse_args()

    # 모델 로드
    predictor = StancePredictor(
        model_path=args.model_path,
        model_name=args.model_name,
        max_length=args.max_length
    )

    # 예측 모드 실행
    if args.interactive:
        # 대화형 모드
        interactive_mode(predictor)

    elif args.text:
        # 단일 텍스트 예측
        result = predictor.predict(args.text, return_probs=True)

        print("\n" + "="*60)
        print("예측 결과")
        print("="*60)
        print(f"텍스트: {args.text}")
        print(f"예측 스탠스: {result['predicted_label_name']} ({result['predicted_label_name_en']})")
        print(f"확신도: {result['confidence']*100:.1f}%")
        print("\n확률 분포:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob*100:.1f}%")
        print("="*60)

    elif args.input_file:
        # CSV 파일 배치 예측
        if not args.output_file:
            # 기본 출력 파일명 생성
            base_name = os.path.splitext(args.input_file)[0]
            args.output_file = f"{base_name}_predicted.csv"

        predictor.predict_csv(
            input_file=args.input_file,
            output_file=args.output_file,
            text_column=args.text_column,
            batch_size=args.batch_size
        )

    else:
        # 인자가 없으면 대화형 모드로 실행
        print("예측 모드가 지정되지 않았습니다. 대화형 모드로 실행합니다.")
        interactive_mode(predictor)


if __name__ == '__main__':
    main()

"""
스탠스 분석 모델
KoBERT를 사용하여 뉴스 기사를 옹호/중립/비판으로 분류
"""

import torch
import torch.nn as nn
from transformers import BertModel

try:
    from kobert_transformers import get_tokenizer

    KOBERT_AVAILABLE = True
except ImportError:
    KOBERT_AVAILABLE = False


class StanceClassifier(nn.Module):
    """
    KoBERT 기반 스탠스 분류 모델

    Classes:
        0: 옹호 (Support)
        1: 중립 (Neutral)
        2: 비판 (Oppose)
    """

    def __init__(self, n_classes=3, dropout=0.3):
        super(StanceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("skt/kobert-base-v1")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """
        순전파

        Args:
            input_ids: 토큰화된 입력 ID
            attention_mask: 어텐션 마스크

        Returns:
            logits: 클래스별 예측 점수
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] 토큰의 출력 사용
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class StancePredictor:
    """스탠스 예측기"""

    STANCE_LABELS = {0: "옹호", 1: "중립", 2: "비판"}

    def __init__(self, model_path=None, device=None):
        """
        초기화

        Args:
            model_path: 학습된 모델 경로 (None이면 새 모델)
            device: 사용할 디바이스 (None이면 자동 선택)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # KoBERT 토크나이저 로드
        if KOBERT_AVAILABLE:
            self.tokenizer = get_tokenizer()
        else:
            from transformers import BertTokenizer

            self.tokenizer = BertTokenizer.from_pretrained("skt/kobert-base-v1")

        self.model = StanceClassifier()

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, max_length=512):
        """
        텍스트의 스탠스 예측

        Args:
            text: 뉴스 기사 텍스트
            max_length: 최대 시퀀스 길이

        Returns:
            dict: {
                'stance': 스탠스 레이블,
                'confidence': 신뢰도,
                'probabilities': 각 클래스별 확률
            }
        """
        # 토큰화
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # 예측
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)

        predicted_class = predicted_class.item()
        confidence = confidence.item()

        return {
            "stance": self.STANCE_LABELS[predicted_class],
            "stance_id": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                self.STANCE_LABELS[i]: round(prob, 4) for i, prob in enumerate(probabilities[0].tolist())
            },
        }

    def predict_batch(self, texts, max_length=512, batch_size=16):
        """
        여러 텍스트의 스탠스를 배치로 예측

        Args:
            texts: 뉴스 기사 텍스트 리스트
            max_length: 최대 시퀀스 길이
            batch_size: 배치 크기

        Returns:
            list: 예측 결과 리스트
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # 배치 토큰화
            encodings = self.tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            # 예측
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                confidences, predicted_classes = torch.max(probabilities, dim=1)

            # 결과 변환
            for j in range(len(batch_texts)):
                predicted_class = predicted_classes[j].item()
                confidence = confidences[j].item()

                results.append(
                    {
                        "stance": self.STANCE_LABELS[predicted_class],
                        "stance_id": predicted_class,
                        "confidence": round(confidence, 4),
                        "probabilities": {
                            self.STANCE_LABELS[k]: round(prob, 4) for k, prob in enumerate(probabilities[j].tolist())
                        },
                    }
                )

        return results

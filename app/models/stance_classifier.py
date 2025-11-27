"""
스탠스 분석 모델
KoBERT를 사용하여 뉴스 기사를 옹호/중립/비판으로 분류
"""

import json
import logging
import os

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, BertModel

logger = logging.getLogger(__name__)


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

    def __init__(self, model_path=None, repo_id=None, device=None):
        """
        초기화

        Args:
            model_path: 학습된 모델 경로 (로컬 파일)
            repo_id: HuggingFace Hub 레포지토리 ID (예: "gaaahee/political-news-stance-classifier")
            device: 사용할 디바이스 (None이면 자동 선택)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {}

        # HuggingFace Hub에서 모델 로드
        if repo_id:
            model_path = self._load_from_hub(repo_id)

        # 토크나이저 로드 (monologg/kobert 사용 - 학습 시 사용한 토크나이저)
        tokenizer_name = self.config.get("tokenizer", "monologg/kobert")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        logger.info(f"Tokenizer loaded: {tokenizer_name}")

        # 모델 초기화
        self.model = StanceClassifier(
            n_classes=self.config.get("num_labels", 3),
            dropout=self.config.get("dropout", 0.3)
        )

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Model loaded from: {model_path}")

        self.model.to(self.device)
        self.model.eval()

    def _load_from_hub(self, repo_id: str) -> str:
        """HuggingFace Hub에서 모델 파일 다운로드"""
        logger.info(f"Loading model from HuggingFace Hub: {repo_id}")

        # config.json 다운로드
        try:
            config_path = hf_hub_download(repo_id, "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            logger.info("Config loaded from Hub")
        except Exception as e:
            logger.warning(f"Could not load config.json: {e}")

        # model.pth 다운로드
        model_path = hf_hub_download(repo_id, "model.pth")
        logger.info(f"Model downloaded to: {model_path}")

        return model_path

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

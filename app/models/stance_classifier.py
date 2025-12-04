"""
스탠스 분석 모델
KoBERT를 사용하여 뉴스 기사를 옹호/중립/비판으로 분류
"""

import json
import logging

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

    def forward(self, input_ids, attention_mask, output_attentions=False):
        """
        순전파

        Args:
            input_ids: 토큰화된 입력 ID
            attention_mask: 어텐션 마스크
            output_attentions: Attention 출력 여부

        Returns:
            logits: 클래스별 예측 점수
            attentions: (optional) Attention 가중치
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )

        # [CLS] 토큰의 출력 사용
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if output_attentions:
            return logits, outputs.attentions
        return logits


class StancePredictor:
    """스탠스 예측기"""

    STANCE_LABELS = {0: "옹호", 1: "중립", 2: "비판"}
    STANCE_THRESHOLD = 0.25  # 스탠스 분류 임계값

    @staticmethod
    def extract_attention_evidence(input_ids, attention_mask, attentions, tokenizer_instance, top_k=5):
        """
        Attention 점수에서 주요 단어 추출

        Args:
            input_ids: 토큰 ID
            attention_mask: 어텐션 마스크
            attentions: BERT attention 출력 (12 layers)
            tokenizer_instance: 토크나이저
            top_k: 상위 몇 개 단어를 추출할지

        Returns:
            dict: {
                'key_words': 주요 단어 리스트,
                'attention_scores': 각 단어의 attention 점수,
                'explanation': 설명 문장
            }
        """
        # 마지막 레이어의 attention 사용 (모든 헤드 평균)
        last_layer_attention = attentions[-1]  # (batch, heads, seq_len, seq_len)

        # [CLS] 토큰이 다른 토큰들에 주는 attention (모든 헤드 평균)
        cls_attention = last_layer_attention[0, :, 0, :].mean(dim=0)  # (seq_len,)

        # 토큰 가져오기
        tokens = tokenizer_instance.convert_ids_to_tokens(input_ids[0])
        actual_length = attention_mask[0].sum().item()

        # 토큰별 attention 점수 (special tokens 제외)
        token_attention_pairs = []
        for i in range(1, int(actual_length) - 1):  # [CLS], [SEP] 제외
            token = tokens[i]
            score = cls_attention[i].item()

            # 특수 토큰, 짧은 토큰 제외
            if token in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
                continue
            if len(token.replace('##', '').replace('▁', '')) < 2:
                continue

            token_attention_pairs.append((token, score))

        # attention 점수 기준 정렬
        token_attention_pairs.sort(key=lambda x: x[1], reverse=True)

        # 상위 K개 추출 및 subword 처리
        key_words = []
        attention_scores = []

        for token, score in token_attention_pairs[:top_k]:
            # subword 마커 제거
            clean_token = token.replace('##', '').replace('▁', '')
            if clean_token and clean_token not in key_words:
                key_words.append(clean_token)
                attention_scores.append(round(score, 4))

        # 설명 생성
        if key_words:
            words_str = "', '".join(key_words[:3])
            explanation = f"모델이 '{words_str}' 등의 표현에 주목하여 판단"
        else:
            explanation = "주목할 만한 특정 표현이 감지되지 않음"

        return {
            'key_words': key_words,
            'attention_scores': attention_scores,
            'explanation': explanation
        }

    @staticmethod
    def classify_stance(support_prob: float, oppose_prob: float) -> tuple:
        """
        스탠스 점수 기반 분류

        Args:
            support_prob: 옹호 확률
            oppose_prob: 비판 확률

        Returns:
            (stance_id, stance_score, stance_clarity)
            - stance_id: 0(옹호), 1(중립), 2(비판)
            - stance_score: P(옹호) - P(비판), 범위 -1 ~ +1
            - stance_clarity: |stance_score|, 0에 가까울수록 중립
        """
        stance_score = support_prob - oppose_prob
        stance_clarity = abs(stance_score)

        if stance_score >= StancePredictor.STANCE_THRESHOLD:
            stance_id = 0  # 옹호
        elif stance_score <= -StancePredictor.STANCE_THRESHOLD:
            stance_id = 2  # 비판
        else:
            stance_id = 1  # 중립

        return stance_id, stance_score, stance_clarity

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
            n_classes=self.config.get("num_labels", 3), dropout=self.config.get("dropout", 0.3)
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
                'stance_id': 스탠스 ID,
                'confidence': 신뢰도,
                'stance_score': 스탠스 점수 (P(옹호) - P(비판)),
                'stance_clarity': 스탠스 명확도 (|stance_score|),
                'probabilities': 각 클래스별 확률,
                'classification_reason': 분류 근거
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

        # 예측 (Attention 포함)
        with torch.no_grad():
            logits, attentions = self.model(input_ids, attention_mask, output_attentions=True)
            probabilities = torch.softmax(logits, dim=1)

        probs = probabilities[0].tolist()
        support_prob = probs[0]
        neutral_prob = probs[1]
        oppose_prob = probs[2]

        # 스탠스 점수 기반 분류
        predicted_class, stance_score, stance_clarity = self.classify_stance(support_prob, oppose_prob)
        confidence = probs[predicted_class]

        # 분류 근거 설명 생성
        if predicted_class == 0:  # 옹호
            reason = f"스탠스 점수 {stance_score:.2f} ≥ {self.STANCE_THRESHOLD} (옹호 확률 {support_prob:.2f}이 비판 확률 {oppose_prob:.2f}보다 {stance_clarity:.2f} 높음)"
        elif predicted_class == 2:  # 비판
            reason = f"스탠스 점수 {stance_score:.2f} ≤ -{self.STANCE_THRESHOLD} (비판 확률 {oppose_prob:.2f}이 옹호 확률 {support_prob:.2f}보다 {stance_clarity:.2f} 높음)"
        else:  # 중립
            reason = f"스탠스 점수 {stance_score:.2f}이 -{self.STANCE_THRESHOLD} ~ {self.STANCE_THRESHOLD} 범위 내 (옹호-비판 차이가 {stance_clarity:.2f}로 명확하지 않음)"

        # Attention 기반 판단 근거 추출
        evidence = self.extract_attention_evidence(input_ids, attention_mask, attentions, self.tokenizer)

        return {
            "stance": self.STANCE_LABELS[predicted_class],
            "stance_id": predicted_class,
            "confidence": round(confidence, 4),
            "stance_score": round(stance_score, 4),
            "stance_clarity": round(stance_clarity, 4),
            "probabilities": {
                self.STANCE_LABELS[i]: round(prob, 4) for i, prob in enumerate(probs)
            },
            "classification_reason": reason,
            "evidence": evidence,
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

            # 예측 (Attention 포함)
            with torch.no_grad():
                logits, attentions = self.model(input_ids, attention_mask, output_attentions=True)
                probabilities = torch.softmax(logits, dim=1)

            # 결과 변환
            for j in range(len(batch_texts)):
                probs = probabilities[j].tolist()
                support_prob = probs[0]
                neutral_prob = probs[1]
                oppose_prob = probs[2]

                # 스탠스 점수 기반 분류
                predicted_class, stance_score, stance_clarity = self.classify_stance(support_prob, oppose_prob)
                confidence = probs[predicted_class]

                # 분류 근거 설명 생성
                if predicted_class == 0:  # 옹호
                    reason = f"스탠스 점수 {stance_score:.2f} ≥ {self.STANCE_THRESHOLD} (옹호-비판 차이: {stance_clarity:.2f})"
                elif predicted_class == 2:  # 비판
                    reason = f"스탠스 점수 {stance_score:.2f} ≤ -{self.STANCE_THRESHOLD} (비판-옹호 차이: {stance_clarity:.2f})"
                else:  # 중립
                    reason = f"스탠스 점수 {stance_score:.2f}이 ±{self.STANCE_THRESHOLD} 범위 내 (차이: {stance_clarity:.2f})"

                # 개별 샘플의 Attention 추출
                single_input_ids = input_ids[j:j+1]
                single_attention_mask = attention_mask[j:j+1]
                single_attentions = tuple(att[j:j+1] for att in attentions)
                evidence = self.extract_attention_evidence(single_input_ids, single_attention_mask, single_attentions, self.tokenizer)

                results.append(
                    {
                        "stance": self.STANCE_LABELS[predicted_class],
                        "stance_id": predicted_class,
                        "confidence": round(confidence, 4),
                        "stance_score": round(stance_score, 4),
                        "stance_clarity": round(stance_clarity, 4),
                        "probabilities": {
                            self.STANCE_LABELS[k]: round(prob, 4) for k, prob in enumerate(probs)
                        },
                        "classification_reason": reason,
                        "evidence": evidence,
                    }
                )

        return results

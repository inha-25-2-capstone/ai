"""
HuggingFace Spaces용 스탠스 분석 API 서버
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, BertModel

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 모델 정의 ====================

class StanceClassifier(nn.Module):
    """KoBERT 기반 스탠스 분류 모델 (Attention 출력 지원)"""

    def __init__(self, n_classes: int = 3, dropout: float = 0.3):
        super(StanceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("skt/kobert-base-v1")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, output_attentions=False):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if output_attentions:
            return logits, outputs.attentions
        return logits


# ==================== 전역 변수 ====================

model: Optional[StanceClassifier] = None
tokenizer = None
device = None
config = {}

STANCE_LABELS = {0: "support", 1: "neutral", 2: "oppose"}
STANCE_LABELS_KR = {0: "옹호", 1: "중립", 2: "비판"}

# 스탠스 분류 임계값
STANCE_THRESHOLD = 0.25  # |옹호 - 비판| >= 0.25 이면 옹호/비판, 아니면 중립


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

    if stance_score >= STANCE_THRESHOLD:
        stance_id = 0  # 옹호
    elif stance_score <= -STANCE_THRESHOLD:
        stance_id = 2  # 비판
    else:
        stance_id = 1  # 중립

    return stance_id, stance_score, stance_clarity


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


# ==================== 모델 로드 ====================

def load_model():
    """HuggingFace Hub에서 모델 로드"""
    global model, tokenizer, device, config

    repo_id = os.getenv("HF_REPO_ID", "gaaahee/political-news-stance-classifier")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model from HuggingFace Hub: {repo_id}")
    logger.info(f"Device: {device}")

    try:
        # config.json 다운로드
        try:
            config_path = hf_hub_download(repo_id, "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info("Config loaded from Hub")
        except Exception as e:
            logger.warning(f"Could not load config.json: {e}")
            config = {}

        # 토크나이저 로드
        tokenizer_name = config.get("tokenizer", "monologg/kobert")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        logger.info(f"Tokenizer loaded: {tokenizer_name}")

        # 모델 다운로드 및 로드
        model_path = hf_hub_download(repo_id, "model.pth")

        model = StanceClassifier(
            n_classes=config.get("num_labels", 3),
            dropout=config.get("dropout", 0.3)
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        logger.info("Model loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


# ==================== FastAPI 앱 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 모델 로드/해제"""
    load_model()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Political News Stance Classifier API",
    description="KoBERT 기반 뉴스 기사 스탠스 분류 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Pydantic 모델 ====================

class PredictRequest(BaseModel):
    text: str = Field(..., description="분석할 텍스트")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., description="분석할 텍스트 리스트")


class EvidenceResponse(BaseModel):
    key_words: List[str] = Field(..., description="모델이 주목한 주요 단어들")
    attention_scores: List[float] = Field(..., description="각 단어의 attention 점수")
    explanation: str = Field(..., description="판단 근거 설명")


class PredictResponse(BaseModel):
    stance: str = Field(..., description="스탠스 (support/neutral/oppose)")
    stance_kr: str = Field(..., description="스탠스 한국어 (옹호/중립/비판)")
    stance_id: int = Field(..., description="스탠스 ID (0/1/2)")
    confidence: float = Field(..., description="신뢰도 (해당 클래스 확률)")
    stance_score: float = Field(..., description="스탠스 점수: P(옹호)-P(비판), 범위 -1~+1")
    stance_clarity: float = Field(..., description="스탠스 명확도: |stance_score|, 0에 가까울수록 중립")
    probabilities: dict = Field(..., description="클래스별 확률")
    classification_reason: str = Field(..., description="분류 기준 설명")
    evidence: EvidenceResponse = Field(..., description="Attention 기반 판단 근거")


# ==================== API 엔드포인트 ====================

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Political News Stance Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    """헬스 체크"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """단일 텍스트 스탠스 예측"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        # 토큰화
        encoding = tokenizer.encode_plus(
            request.text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # 예측 (Attention 포함)
        with torch.no_grad():
            logits, attentions = model(input_ids, attention_mask, output_attentions=True)
            probabilities = torch.softmax(logits, dim=1)

        probs = probabilities[0].tolist()
        support_prob = probs[0]
        neutral_prob = probs[1]
        oppose_prob = probs[2]

        # 스탠스 점수 기반 분류
        predicted_class, stance_score, stance_clarity = classify_stance(support_prob, oppose_prob)
        confidence = probs[predicted_class]

        # 분류 근거 설명 생성
        if predicted_class == 0:  # 옹호
            reason = f"스탠스 점수 {stance_score:.2f} ≥ {STANCE_THRESHOLD} (옹호 확률 {support_prob:.2f}이 비판 확률 {oppose_prob:.2f}보다 {stance_clarity:.2f} 높음)"
        elif predicted_class == 2:  # 비판
            reason = f"스탠스 점수 {stance_score:.2f} ≤ -{STANCE_THRESHOLD} (비판 확률 {oppose_prob:.2f}이 옹호 확률 {support_prob:.2f}보다 {stance_clarity:.2f} 높음)"
        else:  # 중립
            reason = f"스탠스 점수 {stance_score:.2f}이 -{STANCE_THRESHOLD} ~ {STANCE_THRESHOLD} 범위 내 (옹호-비판 차이가 {stance_clarity:.2f}로 명확하지 않음)"

        # Attention 기반 판단 근거 추출
        evidence = extract_attention_evidence(input_ids, attention_mask, attentions, tokenizer)

        return PredictResponse(
            stance=STANCE_LABELS[predicted_class],
            stance_kr=STANCE_LABELS_KR[predicted_class],
            stance_id=predicted_class,
            confidence=round(confidence, 4),
            stance_score=round(stance_score, 4),
            stance_clarity=round(stance_clarity, 4),
            probabilities={
                "support": round(support_prob, 4),
                "neutral": round(neutral_prob, 4),
                "oppose": round(oppose_prob, 4),
            },
            classification_reason=reason,
            evidence=EvidenceResponse(
                key_words=evidence['key_words'],
                attention_scores=evidence['attention_scores'],
                explanation=evidence['explanation']
            ),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictRequest):
    """배치 텍스트 스탠스 예측"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts are required")

    if len(request.texts) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 texts allowed")

    try:
        results = []
        batch_size = 16

        for i in range(0, len(request.texts), batch_size):
            batch_texts = request.texts[i:i + batch_size]

            # 배치 토큰화
            encodings = tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            # 예측 (Attention 포함)
            with torch.no_grad():
                logits, attentions = model(input_ids, attention_mask, output_attentions=True)
                probabilities = torch.softmax(logits, dim=1)

            # 결과 변환
            for j in range(len(batch_texts)):
                probs = probabilities[j].tolist()
                support_prob = probs[0]
                neutral_prob = probs[1]
                oppose_prob = probs[2]

                # 스탠스 점수 기반 분류
                pred_class, stance_score, stance_clarity = classify_stance(support_prob, oppose_prob)
                confidence = probs[pred_class]

                # 분류 근거 설명 생성
                if pred_class == 0:  # 옹호
                    reason = f"스탠스 점수 {stance_score:.2f} ≥ {STANCE_THRESHOLD} (옹호-비판 차이: {stance_clarity:.2f})"
                elif pred_class == 2:  # 비판
                    reason = f"스탠스 점수 {stance_score:.2f} ≤ -{STANCE_THRESHOLD} (비판-옹호 차이: {stance_clarity:.2f})"
                else:  # 중립
                    reason = f"스탠스 점수 {stance_score:.2f}이 ±{STANCE_THRESHOLD} 범위 내 (차이: {stance_clarity:.2f})"

                # 개별 샘플의 Attention 추출
                single_input_ids = input_ids[j:j+1]
                single_attention_mask = attention_mask[j:j+1]
                single_attentions = tuple(att[j:j+1] for att in attentions)
                evidence = extract_attention_evidence(single_input_ids, single_attention_mask, single_attentions, tokenizer)

                results.append({
                    "stance": STANCE_LABELS[pred_class],
                    "stance_kr": STANCE_LABELS_KR[pred_class],
                    "stance_id": pred_class,
                    "confidence": round(confidence, 4),
                    "stance_score": round(stance_score, 4),
                    "stance_clarity": round(stance_clarity, 4),
                    "probabilities": {
                        "support": round(support_prob, 4),
                        "neutral": round(neutral_prob, 4),
                        "oppose": round(oppose_prob, 4),
                    },
                    "classification_reason": reason,
                    "evidence": {
                        "key_words": evidence['key_words'],
                        "attention_scores": evidence['attention_scores'],
                        "explanation": evidence['explanation']
                    },
                })

        return {"results": results}

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

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
    """KoBERT 기반 스탠스 분류 모델"""

    def __init__(self, n_classes: int = 3, dropout: float = 0.3):
        super(StanceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("skt/kobert-base-v1")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


# ==================== 전역 변수 ====================

model: Optional[StanceClassifier] = None
tokenizer = None
device = None
config = {}

STANCE_LABELS = {0: "support", 1: "neutral", 2: "oppose"}
STANCE_LABELS_KR = {0: "옹호", 1: "중립", 2: "비판"}


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


class PredictResponse(BaseModel):
    stance: str = Field(..., description="스탠스 (support/neutral/oppose)")
    stance_kr: str = Field(..., description="스탠스 한국어 (옹호/중립/비판)")
    stance_id: int = Field(..., description="스탠스 ID (0/1/2)")
    confidence: float = Field(..., description="신뢰도")
    probabilities: dict = Field(..., description="클래스별 확률")


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

        # 예측
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)

        predicted_class = predicted_class.item()
        probs = probabilities[0].tolist()

        return PredictResponse(
            stance=STANCE_LABELS[predicted_class],
            stance_kr=STANCE_LABELS_KR[predicted_class],
            stance_id=predicted_class,
            confidence=round(confidence.item(), 4),
            probabilities={
                "support": round(probs[0], 4),
                "neutral": round(probs[1], 4),
                "oppose": round(probs[2], 4),
            },
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

            # 예측
            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                confidences, predicted_classes = torch.max(probabilities, dim=1)

            # 결과 변환
            for j in range(len(batch_texts)):
                pred_class = predicted_classes[j].item()
                probs = probabilities[j].tolist()

                results.append({
                    "stance": STANCE_LABELS[pred_class],
                    "stance_kr": STANCE_LABELS_KR[pred_class],
                    "stance_id": pred_class,
                    "confidence": round(confidences[j].item(), 4),
                    "probabilities": {
                        "support": round(probs[0], 4),
                        "neutral": round(probs[1], 4),
                        "oppose": round(probs[2], 4),
                    },
                })

        return {"results": results}

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

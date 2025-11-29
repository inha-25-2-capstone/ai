---
title: Political News Stance Classifier
emoji: 📰
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Political News Stance Classifier API

KoBERT 기반 뉴스 기사 스탠스 분류 API

## 기능

- 뉴스 기사의 스탠스(옹호/중립/비판) 분석
- 단일 텍스트 분석 및 배치 분석 지원

## API 엔드포인트

### Health Check
```
GET /health
```

### 단일 텍스트 분석
```
POST /predict
Content-Type: application/json

{
    "text": "분석할 뉴스 기사 텍스트"
}
```

### 배치 분석 (최대 50개)
```
POST /predict/batch
Content-Type: application/json

{
    "texts": ["텍스트1", "텍스트2", ...]
}
```

## 응답 형식

```json
{
    "stance": "support",
    "stance_kr": "옹호",
    "stance_id": 0,
    "confidence": 0.85,
    "probabilities": {
        "support": 0.85,
        "neutral": 0.10,
        "oppose": 0.05
    }
}
```

## 스탠스 레이블

| ID | 영문 | 한국어 |
|---|---|---|
| 0 | support | 옹호 |
| 1 | neutral | 중립 |
| 2 | oppose | 비판 |

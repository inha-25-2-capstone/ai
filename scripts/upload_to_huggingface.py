"""
HuggingFace Hub에 스탠스 분류 모델 업로드 스크립트

Usage:
    # 1. HuggingFace 로그인 (처음 한 번만)
    huggingface-cli login

    # 2. 모델 업로드
    python scripts/upload_to_huggingface.py --repo_name "your-username/korean-news-stance-classifier"

    # 3. 특정 모델 파일 지정
    python scripts/upload_to_huggingface.py --model_path saved_models/stance_classifier_20251126_060326.pth
"""

import argparse
import json
import os
import sys
import shutil
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, AutoTokenizer
from huggingface_hub import HfApi, create_repo, upload_folder

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class StanceClassifier(nn.Module):
    """KoBERT 기반 스탠스 분류 모델"""

    def __init__(self, n_classes=3, dropout=0.3, model_name="skt/kobert-base-v1"):
        super(StanceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def create_model_card(metadata: dict, repo_name: str) -> str:
    """모델 카드 (README.md) 생성"""
    return f"""---
language: ko
license: mit
tags:
  - pytorch
  - bert
  - text-classification
  - stance-detection
  - korean
  - news
datasets:
  - custom
metrics:
  - accuracy
model-index:
  - name: {repo_name.split('/')[-1]}
    results:
      - task:
          type: text-classification
          name: Stance Classification
        metrics:
          - type: accuracy
            value: {metadata.get('test_accuracy', 0.916) * 100:.1f}
            name: Test Accuracy
---

# Korean News Stance Classifier (한국어 뉴스 스탠스 분류기)

KoBERT 기반 한국어 정치 뉴스 스탠스(입장) 분류 모델입니다.

## Model Description

- **Base Model**: skt/kobert-base-v1
- **Tokenizer**: monologg/kobert (중요!)
- **Task**: 3-class stance classification (옹호/중립/비판)
- **Language**: Korean

## Performance

- **Test Accuracy**: {metadata.get('test_accuracy', 0.916) * 100:.1f}%
- **Validation Accuracy**: {metadata.get('best_val_accuracy', 0.939) * 100:.1f}%
- **Training Samples**: {metadata.get('data_info', {}).get('train_samples', 5253)}

## Labels

| Label | Korean | English | Description |
|-------|--------|---------|-------------|
| 0 | 옹호 | support | 정부/여당 정책에 우호적 |
| 1 | 중립 | neutral | 객관적 사실 전달 |
| 2 | 비판 | oppose | 정부/여당 정책에 비판적 |

## Usage

```python
import torch
from transformers import AutoTokenizer

# 토크나이저 로드 (반드시 monologg/kobert 사용!)
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

# 모델 로드
model = torch.load("pytorch_model.bin")
# 또는 state_dict 로드
# model.load_state_dict(torch.load("model.pth"))

# 예측
text = "정부의 새 정책이 경제 성장에 크게 기여할 것으로 기대된다"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

with torch.no_grad():
    outputs = model(inputs["input_ids"], inputs["attention_mask"])
    probs = torch.softmax(outputs, dim=1)
    pred = torch.argmax(probs, dim=1).item()

labels = ["옹호", "중립", "비판"]
print(f"Predicted: {{labels[pred]}} ({{probs[0][pred].item()*100:.1f}}%)")
```

## Important Notes

**토크나이저 주의사항**: 이 모델은 `monologg/kobert` 토크나이저로 학습되었습니다.
반드시 동일한 토크나이저를 사용해야 정확한 결과를 얻을 수 있습니다.

```python
# 올바른 사용법
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

# 잘못된 사용법 (결과가 부정확함)
# tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
```

## Training Details

- **Epochs**: {metadata.get('hyperparameters', {}).get('epochs', 16)}
- **Batch Size**: {metadata.get('hyperparameters', {}).get('batch_size', 16)}
- **Learning Rate**: {metadata.get('hyperparameters', {}).get('learning_rate', 2e-05)}
- **Max Length**: {metadata.get('hyperparameters', {}).get('max_length', 512)}
- **Dropout**: {metadata.get('hyperparameters', {}).get('dropout', 0.3)}

## Citation

If you use this model, please cite:

```bibtex
@misc{{korean-news-stance-classifier,
  title={{Korean News Stance Classifier}},
  author={{Politics News Analysis Team}},
  year={{2024}},
  publisher={{HuggingFace}}
}}
```
"""


def prepare_model_files(model_path: str, metadata_path: str, output_dir: str, repo_name: str):
    """HuggingFace 업로드용 파일 준비"""
    os.makedirs(output_dir, exist_ok=True)

    # 메타데이터 로드
    metadata = {}
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    # 1. 모델 파일 복사
    shutil.copy(model_path, os.path.join(output_dir, "model.pth"))
    print(f"[OK] 모델 파일 복사: model.pth")

    # 2. 전체 모델 저장 (추론용)
    print("모델 로딩 중...")
    model = StanceClassifier(n_classes=3, dropout=0.3, model_name="skt/kobert-base-v1")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 전체 모델 저장
    torch.save(model, os.path.join(output_dir, "pytorch_model.bin"))
    print(f"[OK] 전체 모델 저장: pytorch_model.bin")

    # 3. 설정 파일 저장
    config = {
        "model_type": "kobert-stance-classifier",
        "base_model": "skt/kobert-base-v1",
        "tokenizer": "monologg/kobert",
        "num_labels": 3,
        "label2id": {"support": 0, "neutral": 1, "oppose": 2},
        "id2label": {0: "support", 1: "neutral", 2: "oppose"},
        "label_names_kr": ["옹호", "중립", "비판"],
        "max_length": 512,
        "dropout": 0.3,
        "hidden_size": 768,
        "test_accuracy": metadata.get("test_accuracy", 0.916),
        "training_info": metadata
    }

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[OK] 설정 파일 저장: config.json")

    # 4. 모델 카드 생성
    model_card = create_model_card(metadata, repo_name)
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card)
    print(f"[OK] 모델 카드 저장: README.md")

    # 5. 추론 코드 예시 저장
    inference_code = '''"""
스탠스 분류 추론 예시 코드

Usage:
    from inference import StancePredictor

    predictor = StancePredictor("your-username/korean-news-stance-classifier")
    result = predictor.predict("정부의 새 정책이 경제 성장에 기여할 것으로 기대된다")
    print(result)
"""

import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer
from huggingface_hub import hf_hub_download
import json


class StanceClassifier(nn.Module):
    """KoBERT 기반 스탠스 분류 모델"""

    def __init__(self, n_classes=3, dropout=0.3, model_name="skt/kobert-base-v1"):
        super(StanceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


class StancePredictor:
    """HuggingFace Hub에서 모델을 로드하여 스탠스 예측"""

    def __init__(self, repo_id: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.label_names = ["옹호", "중립", "비판"]
        self.label_names_en = ["support", "neutral", "oppose"]

        # 설정 로드
        config_path = hf_hub_download(repo_id, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # 토크나이저 로드 (반드시 monologg/kobert!)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get("tokenizer", "monologg/kobert"),
            trust_remote_code=True
        )

        # 모델 로드
        model_path = hf_hub_download(repo_id, "model.pth")
        self.model = StanceClassifier(
            n_classes=self.config.get("num_labels", 3),
            dropout=self.config.get("dropout", 0.3),
            model_name=self.config.get("base_model", "skt/kobert-base-v1")
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """단일 텍스트 스탠스 예측"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.get("max_length", 512),
            truncation=True,
            padding="max_length"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)[0]
            pred = torch.argmax(probs).item()

        return {
            "stance": self.label_names_en[pred],
            "stance_kr": self.label_names[pred],
            "confidence": round(probs[pred].item(), 4),
            "probabilities": {
                "support": round(probs[0].item(), 4),
                "neutral": round(probs[1].item(), 4),
                "oppose": round(probs[2].item(), 4)
            }
        }

    def predict_batch(self, texts: list, batch_size: int = 16) -> list:
        """배치 텍스트 스탠스 예측"""
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=self.config.get("max_length", 512),
                truncation=True,
                padding="max_length"
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)

            for j in range(len(batch)):
                pred = torch.argmax(probs[j]).item()
                results.append({
                    "stance": self.label_names_en[pred],
                    "stance_kr": self.label_names[pred],
                    "confidence": round(probs[j][pred].item(), 4),
                    "probabilities": {
                        "support": round(probs[j][0].item(), 4),
                        "neutral": round(probs[j][1].item(), 4),
                        "oppose": round(probs[j][2].item(), 4)
                    }
                })

        return results


if __name__ == "__main__":
    # 사용 예시
    predictor = StancePredictor("your-username/korean-news-stance-classifier")

    test_texts = [
        "정부의 새 정책이 경제 성장에 크게 기여할 것으로 기대된다",
        "야당은 졸속 행정이라며 강하게 반발했다",
        "국회에서 법안 심의가 진행되고 있다"
    ]

    for text in test_texts:
        result = predictor.predict(text)
        print(f"텍스트: {text}")
        print(f"결과: {result}")
        print("-" * 50)
'''

    with open(os.path.join(output_dir, "inference.py"), "w", encoding="utf-8") as f:
        f.write(inference_code)
    print(f"[OK] 추론 코드 저장: inference.py")

    return output_dir


def upload_to_hub(output_dir: str, repo_name: str, private: bool = False):
    """HuggingFace Hub에 업로드"""
    api = HfApi()

    # 리포지토리 생성 (이미 있으면 무시)
    try:
        create_repo(repo_name, private=private, exist_ok=True)
        print(f"[OK] 리포지토리 생성/확인: {repo_name}")
    except Exception as e:
        print(f"[WARN] 리포지토리 생성 중 경고: {e}")

    # 파일 업로드
    print(f"\n파일 업로드 중...")
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_name,
        commit_message="Upload Korean News Stance Classifier model"
    )

    print(f"\n{'='*60}")
    print(f"[OK] 업로드 완료!")
    print(f"모델 URL: https://huggingface.co/{repo_name}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="HuggingFace Hub에 스탠스 분류 모델 업로드")

    parser.add_argument(
        "--model_path",
        type=str,
        default="saved_models/stance_classifier_20251117_125133.pth",
        help="업로드할 모델 파일 경로"
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="메타데이터 JSON 파일 경로"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="HuggingFace 리포지토리 이름 (예: username/model-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="비공개 리포지토리로 생성"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="hf_upload_temp",
        help="임시 출력 디렉토리"
    )

    args = parser.parse_args()

    # 메타데이터 경로 자동 추론
    if args.metadata_path is None:
        # 모델 파일명에서 타임스탬프 추출
        model_filename = os.path.basename(args.model_path)
        if "20251126_060326" in model_filename:
            args.metadata_path = "saved_models/metadata_20251126_060326.json"

    print("=" * 60)
    print("HuggingFace Hub 모델 업로드")
    print("=" * 60)
    print(f"모델 파일: {args.model_path}")
    print(f"메타데이터: {args.metadata_path}")
    print(f"리포지토리: {args.repo_name}")
    print(f"비공개: {args.private}")
    print("=" * 60)

    # 모델 파일 확인
    if not os.path.exists(args.model_path):
        print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {args.model_path}")
        return

    # 파일 준비
    print("\n1. 업로드 파일 준비 중...")
    output_dir = prepare_model_files(
        args.model_path,
        args.metadata_path,
        args.output_dir,
        args.repo_name
    )

    # 업로드
    print("\n2. HuggingFace Hub 업로드 중...")
    upload_to_hub(output_dir, args.repo_name, args.private)

    # 임시 파일 정리 여부 안내
    print(f"\n[INFO] 임시 파일은 '{args.output_dir}' 폴더에 있습니다.")
    print("[INFO] 필요 없으면 삭제해도 됩니다.")


if __name__ == "__main__":
    main()

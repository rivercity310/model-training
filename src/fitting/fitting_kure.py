import os
import json
import torch
import random
from tqdm import tqdm
from dataclasses import dataclass
from dotenv import load_dotenv
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from sentence_transformers import SentenceTransformer, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from transformers import get_linear_schedule_with_warmup

# GPU 지원 Pytorch 설치
# https://pytorch.org/get-started/locally/#slide-out-widget-area
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 환경변수 로드
load_dotenv()

# 상수
EMB_TB_NM = "tb_ncs_comp_unit_emb_test"
KURE_DATASET_GLOB = "kure_train_dataset_*.json"
USE_LORA = False
USE_TRAIN = False
BASE_MODEL_NAME = os.getenv("EMBEDDING_MODEL")
IS_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if IS_AVAILABLE else "cpu"

print(f"GPU 사용 가능 여부: {IS_AVAILABLE}")
print(f"GPU 장치 수: {torch.cuda.device_count()}")
print(f"현재 GPU 이름: {torch.cuda.get_device_name(0)}")


@dataclass(frozen=True)
class Paths:
    # 폴더 경로
    ROOT = Path(os.getenv("PROJECT_ROOT_DIR")).resolve()
    CSV = ROOT / "data" / "csv"
    ANALYSIS = ROOT / "data" / "analysis"
    JSON = ROOT / "data" / "json"
    MODEL_OUTPUT = ROOT / "output"
    KURE_DATASET = JSON / "ncs"

    # 파일 경로
    F_EMB_CSV = CSV / f"{EMB_TB_NM}.csv"

    @classmethod
    def get_kure_dataset_json(cls, batch_num: int) -> Path:
        return cls.JSON / "ncs" / f"kure_train_dataset_{batch_num}.json"


# 경로 정의
DATE = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
MODEL_OUTPUT_PATH = Paths.MODEL_OUTPUT / f"kure_finetuned_{DATE}"


if not MODEL_OUTPUT_PATH.exists():
    MODEL_OUTPUT_PATH.mkdir(parents=True)


def load_ncs_dataset(path):
    """JSON 파일을 불러와 모델 학습에 맞는 InputExample 리스트로 병환"""
    train_examples = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            query = item["query"]
            positive_doc = item["positive_document"]
            similarity = float(item["similarity"])
            target = f"{positive_doc['ncs_title']}: {positive_doc['ncs_description']}"
            train_examples.append(InputExample(texts=[query, target], label=similarity))

    return train_examples


class NcsStreamDataset(IterableDataset):
    def __init__(self, filepaths):
        super().__init__()
        self.filepaths = filepaths

    def __iter__(self):
        # 매 에폭마다 파일 순서를 섞어주면 학습 효과 향상
        random.shuffle(self.filepaths)

        # 각 파일 경로에 대해 반복
        for filepath in self.filepaths:
            # 파일에서 샘플들을 로드 (이때는 파일 하나 분량만 메모리에 올라감)
            samples = load_ncs_dataset(filepath)

            # 해당 파일의 각 샘플을 하나씩 반환
            for sample in samples:
                yield sample


def train():
    if not IS_AVAILABLE:
        print("WARN: GPU가 사용 가능하지 않습니다. CPU로 학습을 진행합니다.")

    # 훈련 데이터셋 정의
    # - Streaming 데이터셋은 전체 길이를 미리 알 수 없으므로 warmup과 evaluation step을 고정된 값으로 설정
    epochs = 4
    learning_rate = 2e-5
    train_batch_size = 16
    warmup_steps = 500
    total_steps = 2000

    # 모든 훈련 데이터셋 경로를 메모리에 로딩
    all_filepaths = list(Paths.KURE_DATASET.glob(KURE_DATASET_GLOB))
    random.shuffle(all_filepaths)

    if not all_filepaths:
        raise ValueError("학습 데이터 파일이 없습니다.")

    # 데이터를 훈련용 90%, 검증용 10%로 분할
    train_size = int(len(all_filepaths) * 0.9)
    train_filepaths = all_filepaths[:train_size]
    eval_filepaths = all_filepaths[train_size:]
    print(
        f"[데이터] 전체 파일 수: {len(all_filepaths)}, 훈련 파일 수: {len(train_filepaths)}, 검증 파일 수: {len(eval_filepaths)}"
    )

    # 훈련 데이터용 IterableDataset 인스턴스 생성 -> 큰 훈련 데이터를 스트리밍 방식으로 로드
    train_dataset = NcsStreamDataset(train_filepaths)

    # 검증 데이터는 Evaluator를 위해 메모리에 로드
    # - 검증 데이터는 보통 크기가 작으므로 메모리 부담이 적음
    eval_samples = []
    for filepath in eval_filepaths:
        eval_samples.extend(load_ncs_dataset(filepath))

    if not eval_samples:
        raise ValueError("검증 데이터가 없습니다.")

    print(f"[데이터] 검증 샘플 수: {len(eval_samples)}")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        eval_samples, name="ncs-eval"
    )

    # 모델
    model = SentenceTransformer(BASE_MODEL_NAME)
    print(f"[모델] {BASE_MODEL_NAME} 모델의 학습을 시작합니다.")

    # LoRa 방식으로 학습
    if USE_LORA:
        # - SentenceTransformer 내부 Transformer 모델에 LoRa 적용
        # - 일반적으로 어텐션 레이어의 query, key, value 프로젝션에 적용
        lora_config = LoraConfig(
            r=16,                        # Rank: 8, 16, 32.. -> 높을수록 표현력과 파라미터 수 증가
            lora_alpha=32,               # LoRa Scailing Factor: 일반적으로 Rank * 2
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )

        # 모델에 LoRa 어댑터 추가
        # - SentenceTransformer 모델의 Transformer 부분에 PEFT 모델 적용
        model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)
        print("[LoRa] LoRa 어댑터를 모델에 적용했습니다.")
        model[0].auto_model.print_trainable_parameters()     # 학습 가능한 파라미터 수 출력

        # 모델을 GPU/CPU로 이동
        model.to(DEVICE)

        # v2.x 에서는 DataLoader를 직접 생성
        # DataLoader는 이제 IterableDataset으로부터 실시간으로 데이터를 스트리밍함
        # IterableDataset은 shuffle=True 옵션을 지원하지 않음 (필요 시 Dataset 내부에서 구현)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            collate_fn=model.smart_batching_collate
        )

        # 손실 함수 정의
        # 1. MultipleNegativesRankingLoss
        # - 긍정쌍을 제외한 나머지 (질문, 다른NCS) 쌍들의 유사도는 낮추도록 학습 -> 검색/추천에 매우 효과적
        # 2. CosineSimilarityLoss
        # - 0.95점짜리 쌍은 벡터 공간에서 매우 가깝게, 0.7점짜리 쌍은 적당히 가깝게 배치하도록 학습하여 관계의 '정도'를 학습
        train_loss = losses.CosineSimilarityLoss(model).to(DEVICE)

        # 옵티마이저 정의
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        # IterableDataset은 전체 길이를 알 수 없으므로, 총 스텝 수를 예상하여 스케줄러 설정
        # 예: (파일당 평균 샘플 수 * 파일 수) / 배치 사이즈 * 에폭
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        # --- 수동 훈련 루프 (fit() 대체) ---
        print("수동 훈련 루프를 시작합니다.")
        global_step = 0
        best_score = -1

        for epoch in range(epochs):
            print(f"[Epoch {epoch + 1}/{epochs}]")

            # 모델을 훈련 모드로 설정
            model.train()
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} Training", leave=False)

            for batch in progress_bar:
                sentence_features, labels = batch

                features_on_device = []
                for f in sentence_features:
                    features_on_device.append({key: val.to(DEVICE) for key, val in f.items()})

                labels = labels.to(DEVICE)

                # 손실 계산
                # CosineSimilarityLoss 함수는 모델의 순전파 처리
                loss = train_loss(features_on_device, labels)

                # 역전파 및 가중치 업데이트
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # 진행률 표시줄에 현재 loss 값 표시
                progress_bar.set_postfix(loss=loss.item())

                global_step += 1

                if evaluator is not None and global_step > 0 and global_step % 1000 == 0:
                    print(f"\nStep {global_step}: 모델 성능 평가 중...")
                    model.eval()  # 모델을 평가 모드로 설정

                    # Evaluator를 실행하고 결과를 딕셔너리로 받습니다.
                    score_dict = evaluator(model)

                    # cosine_spearman 점수 추출
                    # - 실제 점수 값보다는 순위가 얼마나 비슷한지 초점
                    current_score = score_dict['ncs-eval_spearman_cosine']

                    # 최고 점수가 갱신되면 LoRa 어댑터만 저장
                    if current_score > best_score:
                        best_score = current_score
                        print(f"새로운 최고 점수 달성: {best_score:.4f}. LoRa 어댑터를 저장합니다.")
                        lora_adapter_path = MODEL_OUTPUT_PATH / "lora_adapter_best"
                        lora_adapter_path.mkdir(parents=True, exist_ok=True)
                        model[0].auto_model.save_pretrained(str(lora_adapter_path))

                    model.train()  # 다시 훈련 모드로 전환

    # Fully Fine-tuning
    else:
        # - 훈련 스텝의 10%를 워밍업으로 사용More actions
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=total_steps,
            warmup_steps=warmup_steps,
            output_path=str(MODEL_OUTPUT_PATH),
            optimizer_params={"lr": learning_rate},
            show_progress_bar=True
        )

    # 모델 학습 실행
    print("🎉 모델 학습이 완료되었습니다.")


if __name__ == "__main__":
    if USE_TRAIN:
        train()

    # 훈련된 모델 테스트
    finetuned_model = SentenceTransformer(BASE_MODEL_NAME)

    if USE_LORA:
        lora_adapter_path = MODEL_OUTPUT_PATH / "lora_adapter"

        # 원본 모델에 LoRa 어댑터 결합
        finetuned_model[0].auto_model = PeftModel.from_pretrained(
            finetuned_model[0].auto_model,
            str(lora_adapter_path)
        )

        print(f"\n[추론] 원본 모델({BASE_MODEL_NAME})에 LoRa 어댑터({lora_adapter_path})를 결합했습니다.")

    # 검색 대상이 될 NCS 직무 정보
    corpus_docs_data = []
    batch_num = 1

    for json_path in Paths.KURE_DATASET.glob(KURE_DATASET_GLOB):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                corpus_docs_data.append(
                    {
                        "code": item["positive_document"]["ncs_code"],
                        "content": f"{item['positive_document']['ncs_title']}: {item['positive_document']['ncs_description']}]",
                    }
                )

    corpus_contents = [doc["content"] for doc in corpus_docs_data]
    print(f"전체 데이터: {len(corpus_contents)}건")

    # Corpus를 벡터로 변환
    corpus_embeddings = finetuned_model.encode(
        corpus_contents, convert_to_tensor=True, show_progress_bar=True
    )

    # 테스트 질의
    while True:
        test_query = input("Query: ")
        test_query_embedding = finetuned_model.encode(
            test_query, convert_to_tensor=True
        )

        # 의미론적 검색
        hits = util.semantic_search(
            test_query_embedding,
            corpus_embeddings,
            top_k=3
        )[0]

        # 결과 출력
        print(f'\n--- 테스트 질의: "{test_query}" ---')
        print("가장 유사한 NCS 직무 TOP 3:")
        for hit in hits:
            doc_index = hit["corpus_id"]
            score = hit["score"]
            ncs_code = corpus_docs_data[doc_index]["code"]
            doc_content = corpus_docs_data[doc_index]["content"]

            print(f"  - NCS 코드: {ncs_code} (유사도: {score:.4f})")
            print(f"    내용: {doc_content}\n")

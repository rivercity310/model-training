import os
import json
import math
import torch
import random
from torch.utils.data import DataLoader, IterableDataset
from sentence_transformers import SentenceTransformer, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime
from src.util import Paths, KURE_DATASET_GLOB

# GPU 지원 Pytorch 설치
# https://pytorch.org/get-started/locally/#slide-out-widget-area
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 경로 정의
DATE = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
MODEL_OUTPUT_PATH = Paths.MODEL_OUTPUT / f"kure_finetuned_{DATE}"

if not MODEL_OUTPUT_PATH.exists():
    MODEL_OUTPUT_PATH.mkdir()

# GPU 정보
is_available = torch.cuda.is_available()
print(f"GPU 사용 가능 여부: {is_available}")

# GPU가 사용 가능하다면, 장치 이름 출력
if is_available:
    print(f"GPU 장치 수: {torch.cuda.device_count()}")
    print(f"현재 GPU 이름: {torch.cuda.get_device_name(0)}")


def load_ncs_dataset(path):
    """ JSON 파일을 불러와 모델 학습에 맞는 InputExample 리스트로 병환 """
    train_examples = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            query = item['query']
            positive_doc = item['positive_document']
            similarity = float(item['similarity'])
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
                print(sample)
                yield sample


def train() -> bool:
    if not is_available:
        print("WARN: GPU가 사용 가능하지 않습니다. CPU로 학습을 진행합니다.")

    # 모든 훈련 데이터셋을 메모리에 로딩
    all_filepaths = sorted(list(Paths.KURE_DATASET.glob(KURE_DATASET_GLOB)))

    if not all_filepaths:
        print("학습 데이터 파일이 없습니다.")
        return False

    # 데이터를 훈련용 90%, 검증용 10%로 분할
    train_size = int(len(all_filepaths) * 0.9)
    train_filepaths = all_filepaths[:train_size]
    eval_filepaths = all_filepaths[train_size:]
    print(f"[데이터] 전체 파일 수: {len(all_filepaths)}, 훈련 파일 수: {len(train_filepaths)}, 검증 파일 수: {len(eval_filepaths)}")

    # 훈련 데이터용 IterableDataset 인스턴스 생성
    train_dataset = NcsStreamDataset(train_filepaths)

    # 검증 데이터는 Evaluator를 위해 메모리에 로드
    # - 검증 데이터는 보통 크기가 작으므로 메모리 부담이 적음
    eval_samples = []
    for filepath in eval_filepaths:
        eval_samples.extend(load_ncs_dataset(filepath))

    if not eval_samples:
        print("검증 데이터가 없습니다. 계속 진행하지만, 모델 성능 검증 및 자동 저장은 수행되지 않습니다.")
        evaluator = None
    else:
        print(f"[데이터] 검증 샘플 수: {len(eval_samples)}")
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_samples, name='ncs-eval')

    # 모델
    model_name = os.getenv("EMBEDDING_MODEL")
    model = SentenceTransformer(model_name)
    print(f"[모델] {model_name} 모델의 학습을 시작합니다.")

    # 훈련 파라미터
    # - Streaming 데이터셋은 전체 길이를 미리 알 수 없으므로 warmup과 evaluation step을 고정된 값으로 설정
    epochs = 4
    learning_rate = 2e-5
    train_batch_size = 16
    warmup_steps = 500
    evaluation_steps = 1000

    # v2.x 에서는 DataLoader를 직접 생성
    # DataLoader는 이제 IterableDataset으로부터 실시간으로 데이터를 스트리밍함
    # IterableDataset은 shuffle=True 옵션을 지원하지 않음 (필요 시 Dataset 내부에서 구현)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size)

    # 손실 함수 정의
    # 1. MultipleNegativesRankingLoss
    # - 긍정쌍을 제외한 나머지 (질문, 다른NCS) 쌍들의 유사도는 낮추도록 학습 -> 검색/추천에 매우 효과적
    # 2. CosineSimilarityLoss
    # - 0.95점짜리 쌍은 벡터 공간에서 매우 가깝게, 0.7점짜리 쌍은 적당히 가깝게 배치하도록 학습하여 관계의 '정도'를 학습
    train_loss = losses.CosineSimilarityLoss(model)

    # 모델 학습 실행
    # - 훈련 스텝의 10%를 워밍업으로 사용
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=str(MODEL_OUTPUT_PATH),
        optimizer_params={"lr": learning_rate},
        show_progress_bar=True
    )

    print("🎉 모델 학습이 완료되었습니다.")

    # 학습된 모델을 저장
    MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_OUTPUT_PATH))
    print(f"모델이 {str(MODEL_OUTPUT_PATH)} 경로에 저장되었습니다.")

    return True


if __name__ == '__main__':
    success = train()

    if not success:
        print("훈련 실패")
        exit(1)

    # 훈련된 모델 테스트
    finetuned_model = SentenceTransformer(str(MODEL_OUTPUT_PATH))

    # 검색 대상이 될 NCS 직무 정보
    corpus_docs_data = []
    batch_num = 1

    while True:
        json_path = Paths.get_kure_dataset_json(batch_num)

        if not json_path.exists():
            break

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                corpus_docs_data.append({
                    "code": item['positive_document']['ncs_code'],
                    "content": f"{item['positive_document']['ncs_title']}: {item['positive_document']['ncs_description']}]"
                })

        corpus_contents = [doc['content'] for doc in corpus_docs_data]

        # Corpus를 벡터로 변환
        corpus_embeddings = finetuned_model.encode(
            corpus_contents,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        # 테스트 질의
        while True:
            test_query = input("Query: ")
            test_query_embedding = finetuned_model.encode(test_query, convert_to_tensor=True)

            # 의미론적 검색
            hits = util.semantic_search(test_query_embedding, corpus_embeddings, top_k=3)[0]

            # 결과 출력
            print(f"\n--- 테스트 질의: \"{test_query}\" ---")
            print("가장 유사한 NCS 직무 TOP 3:")
            for hit in hits:
                doc_index = hit['corpus_id']
                score = hit['score']
                ncs_code = corpus_docs_data[doc_index]['code']
                doc_content = corpus_docs_data[doc_index]['content']

                print(f"  - NCS 코드: {ncs_code} (유사도: {score:.4f})")
                print(f"    내용: {doc_content}\n")

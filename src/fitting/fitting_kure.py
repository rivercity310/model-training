import os
import json
import math
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime

# GPU 지원 Pytorch 설치
# https://pytorch.org/get-started/locally/#slide-out-widget-area
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 환경변수 로드
load_dotenv()

# 경로 정의
DATE = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
ROOT_DIR = Path(os.getenv("PROJECT_ROOT_DIR")).resolve()
DATASET_PATH = ROOT_DIR / "data" / "json" / "kure_train_dataset.json"
MODEL_OUTPUT_PATH = ROOT_DIR / "output" / f"kure_finetuned_{DATE}"

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
            target = f"{positive_doc['ncs_title']}: {positive_doc['ncs_description']}"
            train_examples.append(InputExample(texts=[query, target]))

    return train_examples


def train():
    if not is_available:
        print("WARN: GPU가 사용 가능하지 않습니다. CPU로 학습을 진행합니다.")

    # 데이터셋 준비
    samples = load_ncs_dataset(DATASET_PATH)

    # 데이터를 훈련용 90%, 검증용 10%로 분할
    train_size = int(len(samples) * 0.9)
    train_samples = samples[:train_size]
    eval_samples = samples[train_size:]
    print(f"[데이터] 전체: {len(samples)}, 훈련: {len(train_samples)}, 검증: {len(eval_samples)}")

    # 훈련 파라미터
    epochs = 4
    learning_rate = 2e-5
    warmup_steps = math.ceil(len(train_samples) * epochs * 0.1)
    train_batch_size = 16

    # 모델 및 훈련 데이터셋 로드
    model_name = os.getenv("EMBEDDING_MODEL")
    model = SentenceTransformer(model_name)
    print(f"[모델] {model_name}로 학습을 시작합니다.")

    # v2.x 에서는 DataLoader를 직접 생성
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    # 손실 함수 정의
    # - MultipleNegativesRankingLoss: 한 배치 내에서 [질문, 정답NCS]쌍 (긍정쌍)의
    # - 나머지 (질문, 다른NCS) 쌍들의 유사도는 낮추도록 학습 -> 검색/추천에 매우 효과적
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # EmbeddingSimilarityEvaluator
    # - 매 에폭(epoch)이 끝날 때마다 모델의 성능을 평가하고, 가장 성능이 좋았던 모델을 자동 저장해줌
    eval_sentences1 = [exp.texts[0] for exp in eval_samples]
    eval_sentences2 = [exp.texts[1] for exp in eval_samples]
    eval_similarity = 1.0
    eval_scores = [eval_similarity] * len(eval_samples)

    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_sentences1,
        sentences2=eval_sentences2,
        scores=eval_scores,
        name='ncs-eval'
    )

    # 4. 모델 학습 실행
    # - 훈련 스텝의 10%를 워밍업으로 사용
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=int(len(train_samples) * 0.1),
        warmup_steps=warmup_steps,
        output_path=str(MODEL_OUTPUT_PATH),
        optimizer_params={"lr": learning_rate}
    )


if __name__ == '__main__':
    train()

    # 훈련된 모델 테스트
    finetuned_model = SentenceTransformer(str(MODEL_OUTPUT_PATH))

    # 검색 대상이 될 NCS 직무 정보
    corpus_docs_data = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
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

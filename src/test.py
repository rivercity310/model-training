import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import faiss
import asyncio
import torch
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from sklearn.cluster import KMeans
from sqlalchemy import create_engine


def train():
    model = SentenceTransformer(os.getenv("EMBEDDINGS_MODEL"))

    # Step 2에서 만든 데이터로 InputExample 리스트 생성
    train_examples = [
        InputExample(texts=['파이썬 강의 설명 1', '자료구조 강의 설명 1'], label=1.0),
        InputExample(texts=['파이썬 강의 설명 1', '자바 강의 설명 1'], label=0.0),
    ]

    # EmbeddingSimilarityEvaluator
    # - 매 에폭(epoch)이 끝날 때마다 모델의 성능을 평가하고, 가장 성능이 좋았던 모델을 자동 저장해줌
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(train_examples, name='sts-dev')

    # DataLoader 및 학습 손실 함수 정의
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)

    # 손실 함수 정의
    # - MultipleNegativesRankingLoss: 한 배치 내에서 (질문, 정답NCS) 쌍의 유사도는 높이고,
    # - 나머지 (질문, 다른NCS) 쌍들의 유사도는 낮추도록 학습합니다. 검색/추천에 매우 효과적입니다.
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 4. 모델 학습 실행
    # - 전체 스텝의 10%를 워밍업으로 사용
    epochs = 4
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        output_path='./my-online-edu-recommender'
    )
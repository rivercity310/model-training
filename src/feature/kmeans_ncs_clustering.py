import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import asyncio
import os
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
from src.infrastructure import get_embedding_manager
from src.util import initialize_ncs, Paths, EMB_TB_NM

# DEBUG 모드 유무
DEBUG_MODE = False


def show_training_graph(vectors, random_state: int = 42):
    # KMeans 클러스터링을 위한 K 값 범위 설정
    k_range = range(10, 100, 10)
    inertia_values = []

    for k in tqdm(k_range, desc="엘보우 방법 실행 중..."):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        kmeans.fit(vectors)
        inertia_values.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia_values, marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.title("Elbow Method for K-Means Clustering")
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(Paths.CSV):
        Paths.CSV.mkdir(parents=True)

    # CSV 파일이 없는 경우 DB 정보로 초기화
    if not os.path.exists(Paths.F_EMB_CSV):
        initialize_ncs(EMB_TB_NM).to_csv(Paths.F_EMB_CSV, index=False)

    # CSV 텍스트 형태 데이터 가공 -> Numpy 배열로 변환
    df = pd.read_csv(Paths.F_EMB_CSV)
    EMB_COL_KEY = "embedding_unit_def"
    df[EMB_COL_KEY] = df[EMB_COL_KEY].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=",")
    )
    embedding_vectors = np.stack(df[EMB_COL_KEY].values).astype(np.float32)

    if DEBUG_MODE:
        show_training_graph(embedding_vectors)

    # 모델 훈련
    OPTIMAL_K = 20
    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init="auto")
    kmeans.fit(embedding_vectors)

    # 학습 이후 Cluster ID 열 추가 -> 매번 최신 클러스터링 결과로 갱신
    df["cluster_id"] = kmeans.labels_

    # ---

    user_input = input("사용자 키워드 입력: ")

    # 사용자 입력에 대한 임베딩 생성
    user_embedding = asyncio.run(
        get_embedding_manager().get_avg_query_embedding([user_input])
    )

    # Tensor가 GPU(CUDA) 장치에 할당된 경우 Numpy로 변환하기 위해 cpu 메모리로 옮기는 작업이 필요
    if isinstance(user_embedding, torch.Tensor):
        user_embedding = user_embedding.cpu().numpy().astype(np.float32)
    else:
        user_embedding = np.asarray(user_embedding).astype(np.float32)

    # 훈련된 모델로 예측
    predicted_cluster_id = kmeans.predict(user_embedding.reshape(1, -1))
    print(f"사용자 입력에 대한 클러스터 ID: {predicted_cluster_id}")

    # 해당 클러스터의 데이터프레임 추출
    cluster_df = df[df["cluster_id"] == predicted_cluster_id[0]]
    print(f"\nCluster {predicted_cluster_id[0]}의 데이터:")
    print(cluster_df[["comp_unit_name", "comp_unit_id", "cluster_id"]].head())

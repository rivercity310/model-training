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


if __name__ == "__main__":
    model = SentenceTransformer("./my-online-edu-recommender")






def load_data_from_db_with_embeddings(engine):
    try:
        query = """
            SELECT
                unit.comp_unit_id,
                unit.comp_unit_name,
                unit.comp_unit_def,
                unit.comp_unit_level,
                emb.embedding_unit_def
            FROM tb_ncs_comp_unit_emb emb
            JOIN tb_ncs_comp_unit unit
            ON emb.comp_unit_id = unit.comp_unit_id
            WHERE unit.useflag = 'Y' AND unit.use_ncs = 'Y'
        """

        df = pd.read_sql(query, engine)
        print(f"성공적으로 {len(df)}건의 데이터를 불러왔습니다.")

        # 중요: 텍스트 형태의 임베딩을 실제 Numpy 배열로 변환
        # 예: "[0.1,0.2,...]" -> np.array([0.1, 0.2, ...])
        # DB 저장 형식에 따라 후처리 방식(sep 등)을 조절해야 합니다.
        df['embedding_unit_def'] = df['embedding_unit_def'].apply(
            lambda x: np.fromstring(x.strip('[]'), sep=',')
        )

        return df

    except Exception as e:
        print(f"데이터베이스 처리 중 오류 발생: {e}")
        return None



# --- 1단계: 기반 작업 (샘플 데이터 생성 및 그래프 구축) ---



def build_skill_graph(data: pd.DataFrame, k_neighbors=20, n_probes=10):
    """
    Pandas DataFrame 형태의 NCS 데이터를 기반으로 역량 관계망(그래프) 구축
    FAISS를 사용해서 브루트 포스 방식이 아닌 ANN 기반으로 시간 복잡도 개선
    """

    # 1. DB에서 불러온 임베딩 벡터를 Faiss가 사용 가능한 형태로 변환
    # - (데이터 개수, 임베딩 차원) 형태의 2D Numpy 배열
    # - float32 자료형
    embedding_unit_def = np.array(list(data['embedding_unit_def'])).astype('float32')
    embedding_dimension = embedding_unit_def.shape[1]

    # 2. Faiss 인덱스 생성 및 벡터 추가
    # IndexFlatL2는 가장 기본적인 L2 거리(유클리드 거리) 기반 인덱스입니다.
    # 코사인 유사도와 L2 거리는 정규화된(normalized) 벡터에서는 동일한 순서를 보장합니다.
    quantizer = faiss.IndexFlatIP(embedding_dimension)
    nlist = int(np.sqrt(len(embedding_unit_def)))
    index = faiss.IndexIVFFlat(quantizer, embedding_dimension, nlist, faiss.METRIC_INNER_PRODUCT)

    print("Faiss 인덱스를 학습(training)합니다...")
    index.train(embedding_unit_def)

    print("Faiss 인덱스에 벡터를 추가(add)합니다...")
    index.add(embedding_unit_def)

    # nprobe 파라미터 설정
    index.nprobe = n_probes

    print(f"Faiss 인덱스 생성 완료. 총 {index.ntotal}개의 벡터가 추가되었습니다.")

    # 3. 각 벡터에 대해 가장 가까운 K개의 이웃 검색
    # search(자기 자신을 포함한 모든 벡터, 찾을 이웃의 수 k+1)
    # 자기 자신도 결과에 포함되므로 k+1개 검색
    distances, indices = index.search(embedding_unit_def, k_neighbors + 1)

    # 방향성 있는 그래프(DiGraph) 생성
    G = nx.DiGraph()

    # 그래프에 노드(능력단위) 추가
    for index, row in data.iterrows():
        G.add_node(row['comp_unit_id'], name=row['comp_unit_name'], level=row['comp_unit_level'])

    # 3. 엣지 생성 시, 미리 계산된 임베딩으로 유사도 계산
    for i in tqdm(range(len(data)), desc="엣지 생성 중..."):
        level_i = int(data.loc[i, 'comp_unit_level'])

        for neighbor_idx, dist in zip(indices[i], distances[i]):
            if i == neighbor_idx or neighbor_idx < 0:
                continue

            level_j = int(data.loc[neighbor_idx, 'comp_unit_level'])

            if level_i >= level_j:
                continue

            # NCS 레벨 차이가 클수록 패널티 부여 (우선 순위에서 밀려남)
            level_gap_penalty = (level_j - level_i - 1) * 0.5
            weight = (1 - dist) + level_gap_penalty

            G.add_edge(data.loc[i, 'comp_unit_id'], data.loc[neighbor_idx, 'comp_unit_id'], weight=weight)

    return G



# --- 2단계: 경로 탐색 및 갭 분석 ---

def find_career_path(graph: nx.DiGraph, current_skill_ids: list, target_skill_id: str):
    """
    주어진 그래프에서 현재 역량으로부터 목표 역량까지의 최적 경로와 스킬 갭을 찾습니다.
    """

    best_path = None
    min_path_cost = float('inf')

    # 현재 보유한 여러 스킬 중 어떤 스킬에서 출발하는 것이 가장 효율적인지 탐색
    for start_id in current_skill_ids:
        try:
            # 다익스트라 알고리즘으로 최단 경로(최소 비용 경로) 탐색
            path = nx.dijkstra_path(graph, source=start_id, target=target_skill_id, weight='weight')
            path_cost = nx.dijkstra_path_length(graph, source=start_id, target=target_skill_id, weight='weight')

            if path_cost < min_path_cost:
                min_path_cost = path_cost
                best_path = path

        except nx.NetworkXNoPath:
            # 경로가 없는 경우는 무시
            continue

    if best_path is None:
        return None, None # 경로를 찾지 못한 경우

    # 스킬 갭 = 추천 경로 중 현재 보유 역량을 제외한 나머지
    skill_gap = [skill_id for skill_id in best_path if skill_id not in current_skill_ids]

    return best_path, skill_gap



def get_course_from_node(comp_unit_id, engine):
    # 1. 파라미터화된 쿼리 사용 (SQL Injection 방지 및 구문 오류 해결)
    # - SQL 쿼리문 안에는 플레이스홀더(%(변수명)s)를 사용합니다.
    query = """
    SELECT sbj_nm
    FROM tb_cos_m
    WHERE cos_id IN (SELECT cos_id
    FROM tb_cos_ncs_mat_d mat
    WHERE mat.comp_unit_id = %(comp_id)s
    AND mat.useflag = 'Y'
    AND mat.del_yn = 'N'
    AND mat.match_rate >= 75
    ORDER BY mat.match_rate DESC
    LIMIT 10
    ) AND useflag = 'Y' AND del_yn = 'N'
    """

    # - params 딕셔너리에 실제 값을 전달합니다.
    params = {'comp_id': comp_unit_id}

    # - read_sql 호출 시 engine과 함께 params를 전달합니다.
    df = pd.read_sql(query, engine, params=params)

    # 2. DataFrame의 특정 컬럼 내용을 리스트로 반환 (로직 오류 해결)
    if df.empty:
        return []

    # dict.fromkeys()를 사용하여 순서를 유지하며 중복 제거
    sbj_nm_list = df['sbj_nm'].tolist()
    return list(dict.fromkeys(sbj_nm_list))



if __name__ == "__main__":
    db_url = "postgresql://u_cufit:cufit23$@223.130.152.134:5432/curationdb"
    engine = create_engine(db_url)
    ncs_df = load_data_from_db_with_embeddings(engine)

    # 1. 역량 관계망 그래프 생성
    print("Step 1: 역량 관계망 그래프를 구축합니다...")
    skill_graph = build_skill_graph(ncs_df)
    print(f"그래프 구축 완료! 노드: {skill_graph.number_of_nodes()}개, 엣지: {skill_graph.number_of_edges()}개\n")

    if ncs_df is not None and not ncs_df.empty:
        while True:
            comp_unit_id = input("능력단위 ID: ")
            target_comp_unit_id = input("타겟 ID: ")

            # 2. 사용자 시나리오 설정
            # 시나리오: 현재 나는 어획물을 분류하고 운반하는 역량을 가지고 있어 -> 어류 종자를 생산하는 능력을 키우고 싶어
            # - 시작: 11.어획물 분류 · 운반 (NCS Level 2) <- 내가 현재 가진 역량
            # - 타겟: 05.어류 종자생산 (NCS Level 5) <- 나의 목표 역량
            my_skills = ['2404010411']
            target_skill = '2404020205'

            my_skills = [comp_unit_id]
            target_skill = target_comp_unit_id

            print("Step 2: 최적 경로 및 스킬 갭을 분석합니다...")
            print(f"▶ 현재 보유 역량: {[skill_graph.nodes[s]['name'] for s in my_skills]}")
            print(f"▶ 목표 역량: {skill_graph.nodes[target_skill]['name']}\n")

            # 3. 경로 탐색 및 결과 출력
            recommended_path, skill_gap = find_career_path(skill_graph, my_skills, target_skill)

            if recommended_path:
                print("Step 3: 분석 결과가 나왔습니다!\n")
                print("="*70)
                print("🚀 당신을 위한 추천 성장 로드맵 🚀")
                print("="*70)

                for i, skill_id in enumerate(recommended_path):
                    node = skill_graph.nodes[skill_id]
                    status = "✅ (보유)" if skill_id in my_skills else "🎯 (학습 필요)"
                    print(f"  Step {i+1}. {node['name']} (Level {node['level']}) {status}")

                print("="*70)

                print("\n💡현재 능력 수준에 맞는 강의")
                print("="*70)

                courses = get_course_from_node(comp_unit_id, engine)

                for course in courses:
                    print(f"\t- {course}")

                print("="*70)

                print("\n💡스킬 갭(Skill Gap)")
                print("="*70)

                for skill_id in skill_gap:
                    node = skill_graph.nodes[skill_id]
                    print(f"{node['name']} (Level {node['level']})")
                    courses = get_course_from_node(skill_id, engine)
                    
                    for course in courses:
                        print(f"\t- {course}")

                    print("\n")
                print("="*70)

            else:
                print("분석 실패: 목표 역량까지 도달 가능한 경로를 찾을 수 없습니다.")


class EmbeddingManager:
    def __init__(self, model_name: str = "nlpai-lab/KURE-v1"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, device="cpu")

    async def get_avg_query_embedding(self, queries: list[str]) -> torch.Tensor:
        if not queries:
            raise ValueError("쿼리 텍스트 목록이 비어있습니다.")

        all_query_embeddings: torch.Tensor = await self.embed_async(queries)

        if all_query_embeddings.nelement() == 0:
            raise ValueError("쿼리 텍스트로부터 유효한 임베딩을 생성할 수 없습니다.")

        if len(queries) > 1:
            final_query_embedding_tensor = torch.mean(all_query_embeddings, dim=0, keepdim=True)
        else:
            final_query_embedding_tensor = all_query_embeddings

        return final_query_embedding_tensor

    async def embed_async(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("TEST")

        try:
            # 이벤트 루프 블로킹 방지를 위해 비동기 실행
            embeddings = await asyncio.to_thread(
                self.model.encode,
                texts,
                show_progress_bar=False,
                convert_to_tensor=True
            )

            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings

        except Exception as e:
            raise ValueError("zzzz")


def load_data_from_db_with_embeddings(engine):
    try:
        query = """
            SELECT
            unit.comp_unit_id,
            unit.comp_unit_name,
            unit.comp_unit_def,
            unit.comp_unit_level,
            emb.embedding_unit_def
            FROM tb_ncs_comp_unit_emb emb
            JOIN tb_ncs_comp_unit unit
            ON emb.comp_unit_id = unit.comp_unit_id
            WHERE unit.useflag = 'Y' AND unit.use_ncs = 'Y'
        """

        df = pd.read_sql(query, engine)

        print(f"성공적으로 {len(df)}건의 데이터를 불러왔습니다.")

        # 텍스트 형태의 임베딩을 실제 Numpy 배열로 변환
        # 예: "[0.1,0.2,...]" -> np.array([0.1, 0.2, ...])
        # DB 저장 형식에 따라 후처리 방식(sep 등)을 조절해야 합니다.
        df['embedding_unit_def'] = df['embedding_unit_def'].apply(
            lambda x: np.fromstring(x.strip('[]'), sep=',')
        )

        return df

    except Exception as e:
        print(f"데이터베이스 처리 중 오류 발생: {e}")
        return None


# 엘보우 방법으로 최적의 K 찾기 (몇개의 군집으로 묶을 것인지)

def draw_graph_for_find_better_k(embedding_vectors):
    inertia_values = []
    k_range = range(10, 101, 10)

    print("엘보우 방법을 위한 K-Means를 실행합니다...")

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(embedding_vectors)

        inertia_values.append(kmeans.inertia_)
        print(f"K={k} 완료, Inertia(응집도)={kmeans.inertia_:.2f}")

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia_values, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    db_url = "postgresql://u_cufit:cufit23$@223.130.152.134:5432/curationdb"
    engine = create_engine(db_url)
    ncs_df = load_data_from_db_with_embeddings(engine)

    # Embedding 모델 초기화
    embedding_manager = EmbeddingManager()

    # FAISS 초기화
    # 1. DB에서 불러온 임베딩 벡터를 Faiss가 사용 가능한 형태로 변환
    #   - (데이터 개수, 임베딩 차원) 형태의 2D Numpy 배열
    #   - float32 자료형
    embedding_unit_def = np.array(list(ncs_df['embedding_unit_def'])).astype("float32")
    embedding_dimension = embedding_unit_def.shape[1]

    # 2. Faiss 인덱스 생성 및 벡터 추가
    # IndexFlatL2는 가장 기본적인 L2 거리(유클리드 거리) 기반 인덱스입니다.
    # 코사인 유사도와 L2 거리는 정규화된(normalized) 벡터에서는 동일한 순서를 보장합니다.
    quantizer = faiss.IndexFlatIP(embedding_dimension)
    nlist = int(np.sqrt(len(embedding_unit_def)))
    index = faiss.IndexIVFFlat(quantizer, embedding_dimension, nlist, faiss.METRIC_INNER_PRODUCT)

    print("Faiss 인덱스를 학습(training)합니다...")
    index.train(embedding_unit_def)

    print("Faiss 인덱스에 벡터를 추가(add)합니다...")
    index.add(embedding_unit_def)

    # nprobe 파라미터 설정
    index.nprobe = n_probes

    print(f"Faiss 인덱스 생성 완료. 총 {index.ntotal}개의 벡터가 추가되었습니다.")



    # https://bommbom.tistory.com/entry/%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0%EB%A7%81-%EC%B5%9C%EC%A0%81-%EA%B5%B0%EC%A7%91-%EC%88%98-%EC%97%98%EB%B3%B4%EC%9A%B0-vs-%EC%8B%A4%EB%A3%A8%EC%97%A3-%EA%B8%B0%EB%B2%95
    # draw_graph_for_find_better_k(ncs_emb_vectors)
    OPTIMAL_K = 20
    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init="auto")
    kmeans.fit(embedding_unit_def)

    # 3. 각 데이터 포인트에 할당된 클러스터 레이블(꼬리표) 가져오기
    cluster_labels = kmeans.labels_ # [0, 15, 4, 15, 22, ...] 와 같은 배열

    # 4. 원본 데이터프레임에 'cluster_id' 컬럼으로 추가
    ncs_df['cluster_id'] = cluster_labels
    print("데이터프레임에 클러스터 ID가 추가되었습니다.")
    print(ncs_df[['comp_unit_id', 'cluster_id']].head())

    # ----- TEST -----

    # 1. 사용자 키워드 입력
    target_goal = "파이썬을 활용한 인공지능 데이터 분석 전문가 과정"
    top_k = 10

    # 2. 키워드와 NCS 능력단위 비교
    """1단계: 목표 키워드와 유사한 능력단위 추출"""
    print(f"'{target_goal}'와 유사한 능력단위 {top_k}개를 검색합니다...")

    # 사용자 키워드를 임베딩 벡터로 변환
    query_vector = embedding_manager.encode([target_goal]).astype('float32')

    # Faiss를 사용해 가장 유사한 K개의 능력단위 인덱스를 검색
    _, indices = index.search(query_vector, top_k)

    # 검색된 인덱스에 해당하는 데이터만 추출
    relevant_indices = indices[0]
    relevant_units_df = ncs_df.iloc[relevant_indices].copy().sort_values(by='comp_unit_level')

    print("정렬된 능력단위를 주제(클러스터)별로 그룹화하여 모듈을 구성합니다...")

    # defaultdict를 사용하면 키가 없을 때 자동으로 빈 리스트를 생성해 편리함
    modules = defaultdict(list)

    # DataFrame을 순회하며 cluster_name을 키로 하여 능력단위 정보 추가
    for _, row in relevant_units_df.iterrows():
        module_name = row['cluster_name']
        unit_info = f"{row['comp_unit_name']} (Level {row['comp_unit_level']})"
        modules[module_name].append(unit_info)



    # 4. 최종 결과 출력
    print("\n" + "="*60)
    print(f"🎯 목표: {target_goal}")
    print("💡 AI가 생성한 교육과정 초안입니다.")
    print("="*60)

    # 모듈 이름(클러스터 이름)으로 정렬하여 출력
    for module_name in sorted(generated_curriculum.keys()):
        print(f"\n📘 Module: {module_name}")
        for unit in generated_curriculum[module_name]:
            print(f"   - {unit}")

    print("="*60)
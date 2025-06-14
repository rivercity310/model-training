import numpy as np
import pandas as pd
import networkx as nx
import faiss
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from sqlmodel import create_engine

# 환경변수 로드
load_dotenv()

# DB Engine
url = os.getenv("DATABASE_URL")
engine = create_engine(url=url, echo=True)

# 경로
EMB_TABLE_NAME = "tb_ncs_comp_unit_emb_test"
ROOT_DIR = Path(os.getenv("PROJECT_ROOT_DIR")).resolve()
CSV_DIR_PATH = ROOT_DIR / "data" / "csv"
ANALYSIS_DIR_PATH = ROOT_DIR / "data" / "analysis"
DATA_FILE_PATH = CSV_DIR_PATH / f"{EMB_TABLE_NAME}.csv"


def initialize_csv():
    query = f"""
        SELECT
        unit.comp_unit_id,
        unit.comp_unit_name,
        unit.comp_unit_def,
        unit.comp_unit_level,
        emb.embedding_unit_def
        FROM {EMB_TABLE_NAME} emb
        JOIN tb_ncs_comp_unit unit
        ON emb.comp_unit_id = unit.comp_unit_id
        WHERE unit.useflag = 'Y' AND unit.use_ncs = 'Y'
    """

    df = pd.read_sql(query, engine)
    df.to_csv(DATA_FILE_PATH, index=False, encoding="utf-8-sig")
    print(f"성공적으로 {len(df)}건의 데이터를 불러왔습니다.")


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
    index.train(embedding_unit_def)
    index.add(embedding_unit_def)
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
        return None, None  # 경로를 찾지 못한 경우

    # 스킬 갭 = 추천 경로 중 현재 보유 역량을 제외한 나머지
    skill_gap = [skill_id for skill_id in best_path if skill_id not in current_skill_ids]

    return best_path, skill_gap


def get_course_from_node(comp_unit_id, engine):
    query = """
        SELECT sbj_nm FROM tb_cos_m
        WHERE cos_id IN (
            SELECT cos_id FROM tb_cos_ncs_mat_d mat
            WHERE mat.comp_unit_id = %(comp_id)s
            AND mat.useflag = 'Y'
            AND mat.del_yn = 'N'
            AND mat.match_rate >= 75
            ORDER BY mat.match_rate DESC
            LIMIT 10
        ) 
        AND useflag = 'Y' 
        AND del_yn = 'N'
    """

    params = {'comp_id': comp_unit_id}
    df = pd.read_sql(query, engine, params=params)

    if df.empty:
        return []

    # dict.fromkeys()를 사용하여 순서를 유지하며 중복 제거
    sbj_nm_list = df['sbj_nm'].tolist()
    return list(dict.fromkeys(sbj_nm_list))


if __name__ == "__main__":
    if not os.path.exists(CSV_DIR_PATH):
        CSV_DIR_PATH.mkdir(parents=True)

    # CSV 파일이 없는 경우 DB에서 초기화
    if not os.path.exists(DATA_FILE_PATH):
        initialize_csv()

    # CSV 파일 읽어오고 임베딩 컬럼 가공
    df = pd.read_csv(DATA_FILE_PATH)
    df['comp_unit_id'] = df['comp_unit_id'].astype(str).str.strip()
    df['embedding_unit_def'] = df['embedding_unit_def'].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=',')
    )

    # 결과를 저장할 리스트
    output_lines = []

    # 1. 역량 관계망 그래프 생성
    print("Step 1: 역량 관계망 그래프를 구축합니다...")
    skill_graph = build_skill_graph(df)
    print(f"Graph Generated [Node: {skill_graph.number_of_nodes()}, Edge: {skill_graph.number_of_edges()}]\n")

    if df is not None and not df.empty:
        comp_unit_id = "2001030406" # input("능력단위 ID (쉼표로 구분): ")
        comp_unit_ids = [s.strip() for s in comp_unit_id.split(',')]
        target_comp_unit_id = "2001030407" # input("타겟 ID: ")

        if not all(s_id in skill_graph.nodes for s_id in comp_unit_ids):
            missing_ids = [s_id for s_id in comp_unit_ids if s_id not in skill_graph.nodes]
            print(f"오류: 입력한 능력단위 ID 중 다음 ID(들)이 그래프에 존재하지 않습니다: {', '.join(missing_ids)}. 다시 확인해주세요.")
            exit(0)

        if target_comp_unit_id not in skill_graph.nodes:
            print(f"입력한 타겟 능력단위 ID '{target_comp_unit_id}'가 그래프에 존재하지 않습니다.")
            exit(0)

        # 분석 시작 메시지는 파일에도 포함
        output_lines.append("=" * 70)
        output_lines.append(f"▶ 분석 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output_lines.append("=" * 70)

        output_lines.append("\n[최적 경로 및 스킬 갭 분석]")
        output_lines.append(f"▶ 현재 보유 역량")
        for s in comp_unit_ids:
            output_lines.append(f"\t- {skill_graph.nodes[s]['name']} (Level {skill_graph.nodes[s]['level']})")
        output_lines.append(f"\n▶ 목표 역량")
        output_lines.append(f"\t- {skill_graph.nodes[target_comp_unit_id]['name']} (Level {skill_graph.nodes[target_comp_unit_id]['level']})\n")

        # 3. 경로 탐색 및 결과 출력
        recommended_path, skill_gap = find_career_path(skill_graph, comp_unit_ids, target_comp_unit_id)
        current_lecture_set = set()
        target_lecture_set = set()

        if recommended_path:
            output_lines.append("[분석 결과]")
            output_lines.append("🚀 당신을 위한 추천 성장 로드맵 🚀")

            for i, skill_id in enumerate(recommended_path):
                node = skill_graph.nodes[skill_id]
                status = "✅ (보유)" if skill_id in comp_unit_ids else "🎯 (학습 필요)"
                output_lines.append(f"\t- {i + 1}. {node['name']} (Level {node['level']}) {status}")

            output_lines.append("\n💡현재 보유한 역량 관련 강의")
            courses = get_course_from_node(comp_unit_id, engine)
            for course in courses:
                if course not in current_lecture_set:
                    output_lines.append(f"\t- {course}")
                    current_lecture_set.add(course)

            output_lines.append("\n💡스킬 갭(Skill Gap)")
            for skill_id in skill_gap:
                node = skill_graph.nodes[skill_id]
                output_lines.append(f"{node['name']} (Level {node['level']})")
                courses = get_course_from_node(skill_id, engine)

                for course in courses:
                    if course not in target_lecture_set:
                        output_lines.append(f"\t- {course}")
                        target_lecture_set.add(course)

                output_lines.append("\n")

        else:
            output_lines.append("분석 실패: 목표 역량까지 도달 가능한 경로를 찾을 수 없습니다.")

        # --- 프로그램 종료 시점에 파일 저장 ---
        ANALYSIS_DIR_PATH.mkdir(parents=True, exist_ok=True)  # 결과 저장할 디렉토리 생성

        # 파일명에 현재 날짜와 시간 포함
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = ANALYSIS_DIR_PATH / f"skill_analysis_report_{timestamp}.txt"

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write('\n'.join(output_lines))

        print(f"\n분석 결과가 '{output_filename}' 파일에 성공적으로 저장되었습니다.")

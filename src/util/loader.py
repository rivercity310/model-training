import pandas as pd
import os
from sqlmodel import create_engine
from dotenv import load_dotenv

load_dotenv()

# DB
conn_url = os.getenv("DATABASE_URL")
engine = create_engine(url=conn_url, echo=True)


def initialize_ncs(emb_table: str) -> pd.DataFrame:
    """
    NCs 데이터를 초기화해서 데이터프레임으로 반환
    :param emb_table: 임베딩 테이블명
    :return: 쿼리 조회 결과
    """
    if emb_table not in ["tb_ncs_comp_unit_emb", "tb_ncs_comp_unit_emb_test"]:
        raise ValueError(f"유효하지 않은 테이블명: ${emb_table}")

    query = f"""
        SELECT
            unit.comp_unit_id,
            unit.comp_unit_name,
            unit.comp_unit_def,
            unit.comp_unit_level,
            emb.embedding_unit_def
        FROM {emb_table} emb
        JOIN tb_ncs_comp_unit unit
        ON emb.comp_unit_id = unit.comp_unit_id
        WHERE unit.useflag = 'Y' AND unit.use_ncs = 'Y'
    """

    df = pd.read_sql(query, engine)
    print(f"NCS 데이터 {len(df)}개 로딩 완료")

    return df


def get_courses_from_comp_unit_id(comp_unit_id):
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

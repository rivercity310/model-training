import pandas as pd
import os
import json
import re
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


def parse_json_from_text(text: str) -> dict | None:
    """
    마크다운 코드 블록(```json)으로 감싸인 JSON 문자열을 텍스트에서 추출하고 파싱합니다.

    Args:
        text: JSON 코드 블록을 포함한 전체 텍스트

    Returns:
        파싱된 Python 딕셔너리. 찾지 못하거나 파싱에 실패하면 None을 반환합니다.
    """
    pattern = re.compile(r"```json(.*?)```", re.DOTALL)

    # 텍스트에서 패턴 검색
    match = pattern.search(text)

    if match:
        # 첫 번째 캡처 그룹(괄호 안의 내용)을 가져옵니다.
        # group(0)은 매치된 전체 문자열 "```json...```"
        # group(1)은 첫 번째 괄호 안의 내용 "..."
        json_string = match.group(1)

        # 앞뒤 공백 및 줄바꿈 문자 제거
        json_string = json_string.strip()

        try:
            # 추출된 문자열을 JSON으로 파싱
            parsed_data = json.loads(json_string)
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"오류: 추출된 문자열이 유효한 JSON 형식이 아닙니다. (Error: {e})")
            print(f"추출된 문자열: {json_string}")
            return None
    else:
        print("오류: 텍스트에서 '```json ... ```' 패턴을 찾을 수 없습니다.")
        return None
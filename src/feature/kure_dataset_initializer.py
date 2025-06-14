import pandas as pd
import time
import json
from tqdm import tqdm
from pathlib import Path
from src.util import Paths, EMB_TB_NM, initialize_ncs, parse_json_from_text
from src.infrastructure import model

# 디버그
DEBUG_MODE = True


def invoke(ncs_name, ncs_level, ncs_desc):
    json_format = """
    [
        {
            "input":        // 사용자 질문
            "similarity":   // 사용자 질문과 NCS의 매칭률
        },
        ...
    ]    
    """

    prompt = f"""
    아래 주어진 NCS 직무정보를 사용자의 질문에 매칭시켜야 해.
    사용자는 주로 자신이 궁금한 분야에 대한 강의를 찾기 위해 질문해.
    이를 고려해서 예상되는 질문을 최소 7개 이상 뽑아줘.
    매칭률이 0.5 미만인 질문은 응답에서 제외해줘.
    
    ## NCS 직무정보
    직무명: {ncs_name}
    직무 수준: {ncs_level}
    직무 설명: {ncs_desc}
    
    ## 응답 형식 (JSON)
    {json_format}
    """

    if DEBUG_MODE:
        print(prompt)

    return model.invoke(prompt).text()


# --- 파일 저장 함수 (새로 추가) ---
def save_dataset(filepath: Path, data_to_append: list):
    """
    JSON 파일에 데이터를 이어쓰거나 새로 생성합니다.

    Args:
        filepath (Path): 저장할 파일 경로
        data_to_append (list): 추가할 데이터 (딕셔너리 리스트)
    """
    if not data_to_append:
        return

    try:
        if not filepath.parent.exists():
            filepath.parent.mkdir()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_append, f, ensure_ascii=False, indent=2)

        print(f"\n✅ {len(data_to_append)}개 항목을 '{filepath}'에 성공적으로 저장했습니다.")

    except Exception as e:
        print(f"\n❌ 파일 저장 중 오류 발생: {e}")


if __name__ == "__main__":
    if not Paths.F_EMB_CSV.exists():
        initialize_ncs(EMB_TB_NM).to_csv(Paths.F_EMB_CSV, index=False)

    df = pd.read_csv(Paths.F_EMB_CSV)

    # LLM 호출 제어 및 저장을 위한 변수 초기화
    CALLS_PER_MINUTE_LIMIT = 15
    SAVE_BATCH_SIZE = 50
    SLEEP_INTERVAL = 60 / CALLS_PER_MINUTE_LIMIT  # 호출 사이의 최소 대기 시간

    results_buffer = []
    call_counter = 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="NCS 데이터셋 생성 중"):
        if call_counter > 0:
            time.sleep(SLEEP_INTERVAL)

        code = row['comp_unit_id']
        name = row['comp_unit_name']
        level = row['comp_unit_level']
        desc = row['comp_unit_def']

        res = invoke(name, level, desc)
        call_counter += 1

        json_res = parse_json_from_text(res)

        if json_res is None:
            print(f"WARN: {name}({code}) 직무의 응답을 파싱할 수 없습니다. 건너뜁니다.")
            continue

        if DEBUG_MODE:
            print(json_res)

        # LLM 응답을 최종 저장 형식으로 변환하여 버퍼에 추가
        for val in json_res:
            record = {
                "query": val['input'],
                "positive_document": {
                    "ncs_code": str(code),
                    "ncs_title": str(name),
                    "ncs_description": str(desc),
                    "level": int(level)
                },
                "similarity": float(val['similarity'])
            }

            results_buffer.append(record)

        # 3. 100번 호출마다 파일에 저장 (Batch Saving)
        if call_counter % SAVE_BATCH_SIZE == 0:
            batch_num = call_counter // SAVE_BATCH_SIZE
            filepath = Paths.get_kure_dataset_json(batch_num)
            save_dataset(filepath, results_buffer)
            results_buffer.clear()

    if results_buffer:
        batch_num = (call_counter - 1) // SAVE_BATCH_SIZE + 1
        filepath = Paths.get_kure_dataset_json(batch_num)
        save_dataset(filepath, results_buffer)
        results_buffer.clear()

    print("\n🎉 모든 작업이 완료되었습니다.")
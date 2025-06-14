import pandas as pd
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.output_parser import StrOutputParser
from src.util import (
    Paths,
    EMB_TB_NM,
    KURE_DATASET_GLOB,
    initialize_ncs,
    parse_json_from_text,
)

# 환경변수 로드
load_dotenv()

# 환경변수
API_KEY = os.getenv("LLM_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL_NAME")
MODEL_PROVIDER = os.getenv("LLM_MODEL_PROVIDER")
IS_GEMINI = MODEL_PROVIDER == "google_genai"

# API 키 세팅
if "google" in MODEL_PROVIDER and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = API_KEY

elif "openai" in MODEL_PROVIDER and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = API_KEY


# --- Pydantic 모델 정의 ---
class Question(BaseModel):
    input: str = Field(description="NCS 직무와 관련된 예상 사용자 질문")
    similarity: float = Field(
        description="예상 질문과 NCS 직무 정보 간의 의미적 유사도 점수 (0.5에서 1.0 사이)"
    )


class GeneratedQuestionList(BaseModel):
    questions: list[Question] = Field(description="생성된 질문 객체들의 리스트")


class NCSDatasetGenerator:
    def __init__(self):
        _chain, _parser = self._create_chain()
        self.chain = _chain
        self.parser = _parser

    @staticmethod
    def _create_chain():
        """LangChain 구성 요소 세팅 및 완성된 체인 생성"""
        print("LangChain 체인을 초기화합니다...")

        # 모델 초기화
        model = init_chat_model(model=MODEL_NAME, model_provider=MODEL_PROVIDER)

        # 파서 초기화
        parser = PydanticOutputParser(pydantic_object=GeneratedQuestionList)

        # 프롬프트 템플릿 생성 (포맷 지침 포함)
        format_instructions = parser.get_format_instructions()

        template_string = """
        당신은 사용자의 잠재적 질문을 생성하는 전문가입니다.
        아래 주어진 NCS 직무정보를 보고, 사용자들이 어떤 질문을 할지 예측해야 합니다.
        사용자는 주로 자신이 궁금한 분야에 대한 강의를 찾기 위해 질문하므로, 학습 및 직무 역량과 관련된 질문을 생성해주세요.
        최소 7개 이상의 다양한 질문을 생성하고, 각 질문과 NCS 직무 간의 예상 유사도 점수를 함께 제공해주세요.
        유사도 점수가 0.5 미만인 질문은 생성하지 마세요.

        ## NCS 직무정보
        - 직무명: {ncs_name}
        - 직무 수준: {ncs_level}
        - 직무 설명: {ncs_desc}

        ## 출력 형식 지침
        {format_instructions}
        """

        prompt_template = PromptTemplate(
            template=template_string,
            input_variables=["ncs_name", "ncs_level", "ncs_desc"],
            partial_variables={"format_instructions": format_instructions},
        )

        return prompt_template | model | StrOutputParser(), parser

    def generate_for_ncs_item(self, ncs_name: str, ncs_level: int, ncs_desc: str):
        """단일 NCS 항목에 대해 질문 리스트 생성"""
        try:
            response_text = self.chain.invoke(
                {"ncs_name": ncs_name, "ncs_level": ncs_level, "ncs_desc": ncs_desc}
            )

            json_string = parse_json_from_text(response_text)

            if json_string:
                return self.parser.parse(json_string)
            else:
                print(f"WARN: 응답에서 JSON을 찾지 못했습니다. (직무: {ncs_name})")
                return None
        except Exception as e:
            print(f"ERROR: 체인 실행 또는 파싱 중 오류 발생. (직무: {ncs_name}) - {e}")
            return None


def save_dataset(filepath: Path, data_to_append: list):
    """
    JSON 파일에 데이터를 이어쓰거나 새로 생성

    Args:
        filepath (Path): 저장할 파일 경로
        data_to_append (list): 추가할 데이터 (딕셔너리 리스트)
    """
    if not data_to_append:
        return

    try:
        if not filepath.parent.exists():
            filepath.parent.mkdir()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_to_append, f, ensure_ascii=False, indent=2)

        print(
            f"\n✅ {len(data_to_append)}개 항목을 '{filepath}'에 성공적으로 저장했습니다."
        )

    except Exception as e:
        print(f"\n❌ 파일 저장 중 오류 발생: {e}")


def get_processed_ids() -> set[str]:
    """출력 디렉토리를 스캔하여 이미 처리된 NCS 코드 ID들을 set으로 반환"""
    output_dir = Paths.KURE_DATASET
    processed_ids = set()

    if not output_dir.exists():
        return processed_ids

    # kure_dataset_*.json 패턴을 가진 모든 파일을 찾습니다.
    for filepath in output_dir.glob(KURE_DATASET_GLOB):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    if (
                        "positive_document" in item
                        and "ncs_code" in item["positive_document"]
                    ):
                        processed_ids.add(item["positive_document"]["ncs_code"])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"WARN: '{filepath}' 파일 처리 중 오류 발생. 건너뜁니다. ({e})")
            continue

    return processed_ids


def get_batch_num() -> int:
    return len(list(Paths.KURE_DATASET.glob(KURE_DATASET_GLOB))) + 1


def process_row(row):
    """DataFrame의 한 행을 받아 LLM을 호출하고 결과를 파싱하여 반환합니다."""
    code = row["comp_unit_id"]
    name = row["comp_unit_name"]
    level = row["comp_unit_level"]
    desc = row["comp_unit_def"]

    results = ncs_dataset_generator.generate_for_ncs_item(name, level, desc)

    if results is None:
        print(f"WARN: {name}({code}) 직무의 응답을 파싱할 수 없습니다. 건너뜁니다.")
        return None

    records = []
    for question in results.questions:
        record = {
            "query": question.input,
            "positive_document": {
                "ncs_code": str(code),
                "ncs_title": str(name),
                "ncs_description": str(desc),
                "level": int(level),
            },
            "similarity": question.similarity,
        }
        records.append(record)

    return records


if __name__ == "__main__":
    print(f"모델 공급자: {MODEL_PROVIDER}")
    ncs_dataset_generator = NCSDatasetGenerator()

    if not Paths.F_EMB_CSV.exists():
        initialize_ncs(EMB_TB_NM).to_csv(Paths.F_EMB_CSV, index=False)

    df = pd.read_csv(Paths.F_EMB_CSV)

    # 이미 처리된 항목들 ID 로드
    processed_ids = get_processed_ids()
    if processed_ids:
        print(
            f"INFO: 기존에 처리된 {len(processed_ids)}개의 항목을 발견했습니다. 이어서 작업을 시작합니다."
        )

    # 처리되지 않은 데이터만 필터링
    unprocessed_df = df[~df["comp_unit_id"].astype(str).isin(processed_ids)].copy()

    if unprocessed_df.empty:
        print("🎉 모든 항목이 이미 처리되었습니다. 작업을 종료합니다.")
        exit()

    print(f"총 {len(df)}개 항목 중, 새로 처리할 항목은 {len(unprocessed_df)}개 입니다.")

    # LLM 호출 제어 및 저장을 위한 변수 초기화
    CALLS_PER_MINUTE_LIMIT = 15
    SAVE_BATCH_SIZE = 50
    SLEEP_INTERVAL = 60 / CALLS_PER_MINUTE_LIMIT  # 호출 사이의 최소 대기 시간
    MAX_WORKERS = 1 if IS_GEMINI else 10

    results_buffer = []
    call_counter = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for index, row in unprocessed_df.iterrows():
            future = executor.submit(process_row, row)
            futures.append(future)

            if IS_GEMINI:
                time.sleep(SLEEP_INTERVAL)

        print(f"{len(futures)}개의 작업을 모두 제출했습니다. 이제 결과를 기다립니다...")

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="NCS 데이터셋 생성 중"
        ):
            records = future.result()

            if records is None:
                continue

            print(records)

            results_buffer.extend(records)
            call_counter += 1

            # N번 호출마다 파일에 저장
            if call_counter > 0 and call_counter % SAVE_BATCH_SIZE == 0:
                filepath = Paths.get_kure_dataset_json(get_batch_num())
                save_dataset(filepath, results_buffer)
                results_buffer.clear()

    # 루프가 끝난 후 버퍼에 남은 데이터가 있으면 마지막으로 저장
    if results_buffer:
        filepath = Paths.get_kure_dataset_json(get_batch_num())
        save_dataset(filepath, results_buffer)
        results_buffer.clear()

    print("\n🎉 모든 작업이 완료되었습니다.")

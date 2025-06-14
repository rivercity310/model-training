import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from src.util import parse_json_from_text

# 환경변수 로드
load_dotenv()

# 환경변수 값 가져오기
API_KEY = os.getenv("LLM_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL_NAME")
MODEL_PROVIDER = os.getenv("LLM_MODEL_PROVIDER")

# API 키 세팅
if "google" in MODEL_PROVIDER and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = API_KEY

elif "openai" in MODEL_PROVIDER and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = API_KEY


# --- Pydantic 모델 정의 ---
class Question(BaseModel):
    input: str = Field(description="NCS 직무와 관련된 예상 사용자 질문")
    similarity: float = Field(description="예상 질문과 NCS 직무 정보 간의 의미적 유사도 점수 (0.5에서 1.0 사이)")


class GeneratedQuestionList(BaseModel):
    questions: list[Question] = Field(description="생성된 질문 객체들의 리스트")


# 모델 초기화
model = init_chat_model(
    model=MODEL_NAME,
    model_provider=MODEL_PROVIDER
)

parser = PydanticOutputParser(pydantic_object=GeneratedQuestionList)
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
    partial_variables={"format_instructions": format_instructions}
)

# 프롬프트, 모델, 파서를 파이프로 연결하여 체인 생성
chain = prompt_template | model | StrOutputParser()


def invoke_ncs_datasest(ncs_name: str, ncs_level: int, ncs_desc: str):
    try:
        # 체인에 입력값을 딕셔너리 형태로 전달하여 실행
        result = chain.invoke({
            "ncs_name": ncs_name,
            "ncs_level": ncs_level,
            "ncs_desc": ncs_desc
        })

        json_string = parse_json_from_text(result)
        print(json_string)

        if json_string:
            parsed_result = parser.parse(json_string)
            return parsed_result
        else:
            print(f"WARN: 응답 텍스트에서 JSON 블록을 찾지 못했습니다.")
            return None

    except Exception as e:
        print(f"LangChain 호출 중 오류 발생: {e}")
        return None
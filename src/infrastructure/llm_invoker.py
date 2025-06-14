import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

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

# 모델 초기화
model = init_chat_model(
    model=MODEL_NAME,
    model_provider=MODEL_PROVIDER
)


def invoke(prompt: str) -> str:
    return model.invoke(prompt).text()


if __name__ == "__main__":
    print(invoke("Hello, how are you?"))

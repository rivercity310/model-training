import asyncio
import torch
import os
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class EmbeddingManager:
    def __init__(self, model_name: str = os.getenv("EMBEDDING_MODEL")):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    async def get_avg_query_embedding(self, queries: list[str]) -> torch.Tensor:
        """
        주어진 쿼리 목록으로부터 최종 검색에 사용할 쿼리 임베딩 평균값 생성
        """
        if not queries:
            raise ValueError("쿼리 텍스트 목록이 비어있습니다.")

        # 모든 텍스트에 대한 임베딩 생성
        all_query_embeddings: torch.Tensor = await self.embed_async(queries)

        if all_query_embeddings.nelement() == 0:
            # embed_async가 빈 텐서를 반환하는 경우 (예: 모든 텍스트가 유효하지 않아 임베딩 실패)
            raise ValueError("쿼리 텍스트로부터 유효한 임베딩을 생성할 수 없습니다.")

        # 쿼리 텍스트가 여러 개면 산술 평균 계산
        # 쿼리 텍스트가 하나인 경우, 해당 임베딩을 그대로 사용
        if len(queries) > 1:
            # (N, D) 형태의 텐서에서 N개 벡터의 평균을 구함 -> (D,) 형태의 벡터
            final_query_embedding_tensor = torch.mean(
                all_query_embeddings, dim=0, keepdim=True
            )
        else:
            final_query_embedding_tensor = all_query_embeddings

        return final_query_embedding_tensor

    async def embed_async(self, texts: list[str]) -> torch.Tensor:
        """
        텍스트 목록에 대한 임베딩을 비동기적으로 생성.

        Args:
            texts: 임베딩할 문자열 목록.

        Returns:
            torch.Tensor: shape=(n, d), 각 벡터는 L2 정규화되어 있음 (unit vector)

        Raises:
            ValueError: 텍스트 수가 MAX_REQUEST_SIZE를 초과하거나 임베딩 실패 시 발생.
        """
        try:
            # 이벤트 루프 블로킹 방지를 위해 비동기 실행
            embeddings: torch.Tensor = await asyncio.to_thread(
                self.model.encode,
                texts,
                show_progress_bar=False,
                convert_to_tensor=True
            )

            # 임베딩 결과를 각각을 정규화해서 반환
            # FAISS에서 코사인 유사도 검색 시 벡터 정규화 필수
            # 정규화 (벡터 정규화: L2 norm)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings

        except Exception as e:
            raise ValueError() from e


@lru_cache(maxsize=1)
def get_embedding_manager() -> EmbeddingManager:
    """
    EmbeddingManager 싱글턴 인스턴스 반환.
    """
    return EmbeddingManager()
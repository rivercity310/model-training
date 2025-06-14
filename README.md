## 주요 기술스택
- Pytorch
- LangChain
- SentenceTransformer
- Faiss

## 시스템 요구사항

- Python 3.12+
- CUDA 지원 GPU (권장)
- NVIDIA 드라이버 및 CUDA 툴킷 (GPU 사용 시)

## 설치 가이드

### 1. Conda 환경 설정

```bash
# Conda 환경 생성 (Python 3.12)
conda create -n model-training python=3.12 -y

# 환경 활성화
conda activate model-training
```

### 2. 의존성 설치

```bash
# requirements.txt 기반 패키지 설치
pip install -r requirements.txt
```

### 3. 환경 변수 설정

`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```
PROJECT_ROOT_DIR=.
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
DATABASE_URL=postgresql://username:password@localhost:5432/your_database
```


## GPU 가속 확인

CUDA가 제대로 설정되었는지 확인하려면 다음 명령어를 실행하세요:

```python
import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"CUDA 버전: {torch.version.cuda}")
print(f"사용 가능한 GPU: {torch.cuda.device_count()}개")
print(f"GPU 이름: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

## 문제 해결

1. **CUDA 사용 불가능한 경우**:
   ```
   nvidia-smi  # GPU 상태 확인
   conda list cudatoolkit  # CUDA 툴킷 설치 확인
   ```

2. **버전 호환성 문제**:
   - PyTorch, CUDA, cuDNN 버전이 호환되는지 확인하세요.
   - [PyTorch 공식 문서](https://pytorch.org/get-started/previous-versions/)에서 호환되는 버전 조합을 확인하세요.
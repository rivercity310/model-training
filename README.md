## NCS 기반 모델 학습 데이터셋 자동화 및 분석 시스템
본 프로젝트는 방대한 NCS(국가직무능력표준) 직무 데이터를 AI 모델 학습에 효과적으로 활용하기 위한 통합 파이프라인 시스템입니다. 최신 LLM을 통해 학습 데이터셋 생성을 자동화하고, 임베딩된 데이터를 심도 있게 분석 및 시각화하여 모델의 성능을 극대화하는 것을 목표로 합니다.



## ✨ 주요 기능
### 1. 자동 데이터셋 생성 및 라벨링
LangChain 프레임워크를 통해 최신 LLM(GPT, Gemini 등)과 상호작용하여, 복잡한 NCS 직무 정보로부터 모델 학습에 필요한 고품질의 '질문-유사도' 데이터셋을 자동으로 구축합니다. 이 과정은 수작업 라벨링에 소요되는 시간과 비용을 획기적으로 절감합니다.

### 2. 고차원 데이터의 군집화 및 인사이트 도출
Sentence-Transformers로 임베딩된 고차원 벡터 데이터셋에 KMeans 클러스터링을 적용합니다. 이를 통해 NCS 직무 데이터의 숨겨진 구조와 패턴을 발견하고, 각 직무 그룹의 의미적 특성을 정량적으로 분석할 수 있습니다.

### 3. 초고속 유사도 검색 및 관계 시각화
Facebook AI의 Faiss 라이브러리를 활용하여 대규모 데이터셋에서도 실시간에 가까운 속도로 유사도 검색(ANN)을 수행합니다. 이를 바탕으로 직무 간의 관계를 네트워크 그래프로 시각화하여, 직관적인 커리큘럼 탐색이나 직무 추천 시스템의 기반을 마련합니다.

### 4. SBERT 모델 파인튜닝 지원
자동 생성된 데이터셋을 활용하여 Sentence-BERT(SBERT) 기반의 언어 모델을 특정 도메인에 맞게 파인튜닝하는 기능을 지원합니다. 이를 통해 NCS 직무 검색 및 매칭 정확도를 비약적으로 향상시킬 수 있습니다.


## 🛠️ 주요 기술 스택
- AI & Data: PyTorch, LangChain, Sentence-Transformers, Faiss, scikit-learn
- Data Handling: Pandas, NumPy
- Environment: Python, Conda

## ⚙️ 시스템 요구사항
- Python 3.12+
- NVIDIA GPU (CUDA 지원) - 학습 및 Faiss 인덱싱에 적극 권장
- NVIDIA 드라이버 및 CUDA Toolkit (GPU 사용 시)

## 🚀 설치 및 설정 가이드
### 1. Conda 가상환경 생성 및 활성화
프로젝트의 독립적인 실행 환경을 보장하기 위해 Conda 가상환경 생성을 권장합니다. 터미널을 열고 아래 명령어를 실행해주세요.


```bash
# Conda 가상환경 생성 (Python 3.12)
conda create -n {env-name} python=3.12 -y

# 생성된 환경 활성화
conda activate {env-name}
````

### 2. 의존성 설치 및 환경변수 설정
활성화된 환경에서 필요한 라이브러리를 설치하고, 프로젝트 실행에 필요한 환경 변수를 구성합니다.

```bash
pip install -r requirements.txt
```

이후, 프로젝트의 루트 디렉터리에 .env 파일을 생성하고 아래 내용을 참고하여 본인의 환경에 맞게 값을 입력해주세요.
```bash
# 프로젝트 루트 디렉터리 경로 (필요시 절대 경로 지정)
PROJECT_ROOT_DIR=/path/to/your/project

# Sentence Transformer 임베딩 모델
EMBEDDING_MODEL=nlpai-lab/KURE-v1

# 데이터베이스 연결 정보 (필요시)
DATABASE_URL=postgresql://user:password@host:port/dbname

# LLM API 설정
LLM_API_KEY="sk-..." # 또는 "gsk-..."
LLM_MODEL_NAME="gpt-4o-mini" # 또는 "gemini-2.0-flash"
LLM_MODEL_PROVIDER="openai" # 또는 "google"
```

## ▶️ 사용법 예시
프로젝트의 각 기능은 다음 스크립트를 통해 실행할 수 있습니다. 

```bash
# 1. LLM을 이용한 데이터셋 자동 생성 실행
python src/feature/kure_dataset_initializer.py

# 2. NCS 기반 데이터를 KMeans 클러스터링
python src/feature/kmeans_ncs_clustering.py

# 3. Faiss를 이용한 ANN 인덱스 빌드 및 그래프 생성
python src/feature/ncs_curriculum.py

# 4. 임베딩 모델 파인튜닝
python src/fitting/fitting_kure.py
```

## 🩺 GPU 설정 확인 및 문제 해결
설치가 완료된 후, 아래 스크립트를 실행하여 PyTorch가 GPU를 올바르게 인식하는지 확인해 보세요.

```Python
import torch

if torch.cuda.is_available():
    print(f"✅ CUDA 사용 가능 (PyTorch 버전: {torch.__version__})")
    print(f"   - CUDA 버전: {torch.version.cuda}")
    print(f"   - 사용 가능한 GPU: {torch.cuda.device_count()}개")
    print(f"   - GPU 이름: {torch.cuda.get_device_name(0)}")
else:
    print("❌ CUDA를 사용할 수 없습니다. CPU로 실행됩니다.")
```

### 1. CUDA 사용 가능이 False로 나올 경우
- 먼저 터미널에서 nvidia-smi 명령을 실행하여 시스템에 설치된 NVIDIA 드라이버의 상태와 지원하는 CUDA 최고 버전 확인
- conda list cudatoolkit 명령을 통해 현재 Conda 환경에 설치된 CUDA 툴킷 버전을 확인 
- 만약 설치되어 있지 않거나 버전이 맞지 않다면, PyTorch를 GPU 지원 버전으로 재설치

### 2. 버전 호환성 문제
- PyTorch, CUDA, cuDNN, NVIDIA 드라이버는 서로 밀접하게 연관되어 있어 버전 호환성이 매우 중요
- PyTorch 공식 홈페이지를 방문하여 현재 드라이버가 지원하는 CUDA 버전에 맞는 PyTorch 설치 명령어를 직접 확인하고 실행
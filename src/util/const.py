import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

# 환경변수 로드
load_dotenv()

# 상수
EMB_TB_NM = "tb_ncs_comp_unit_emb_test"
KURE_DATASET_GLOB = "kure_train_dataset_*.json"


@dataclass(frozen=True)
class Paths:
    # 폴더 경로
    ROOT = Path(os.getenv("PROJECT_ROOT_DIR")).resolve()
    CSV = ROOT / "data" / "csv"
    ANALYSIS = ROOT / "data" / "analysis"
    JSON = ROOT / "data" / "json"
    MODEL_OUTPUT = ROOT / "output"
    KURE_DATASET = JSON / "ncs"

    # 파일 경로
    F_EMB_CSV = CSV / f"{EMB_TB_NM}.csv"

    @classmethod
    def get_kure_dataset_json(cls, batch_num: int) -> Path:
        return cls.JSON / "ncs" / f"kure_train_dataset_{batch_num}.json"

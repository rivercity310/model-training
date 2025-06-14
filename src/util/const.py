import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

# 환경변수 로드
load_dotenv()

# 상수
EMB_TB_NM = "tb_ncs_comp_unit_emb"


@dataclass(frozen=True)
class Paths:
    # 폴더 경로
    ROOT = Path(os.getenv("PROJECT_ROOT_DIR")).resolve()
    CSV = ROOT / "data" / "csv"
    ANALYSIS = ROOT / "data" / "analysis"
    JSON = ROOT / "data" / "json"

    # 파일 경로
    F_EMB_CSV = CSV / f"{EMB_TB_NM}.csv"
    F_KURE_DS_JSON = JSON / "kure_train_dataset.json"
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from src.feature import initialize_csv

# 환경변수 로드
load_dotenv()

# 경로 정의
ROOT_DIR = Path(os.getenv("PROJECT_ROOT_DIR")).resolve()
DATASET_PATH = ROOT_DIR / "data" / "json" / "kure_train_dataset.json"
CSV_DIR_PATH = ROOT_DIR / "data" / "csv"
CSV_FILE_PATH = CSV_DIR_PATH / "tb_ncs_comp_unit_emb_test.csv"

if not os.path.exists(CSV_DIR_PATH):
    CSV_DIR_PATH.mkdir(parents=True)

# CSV 파일이 없는 경우 DB에서 초기화
if not os.path.exists(DATA_FILE_PATH):
    initialize_csv(EMB_TABLE_NAME)

if __name__ == "__main__":
    pass
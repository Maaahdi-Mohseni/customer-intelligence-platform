from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

data_dir = project_root / "data"

raw_data_dir = data_dir / "raw"

processed_data_dir = data_dir / "processed"

dataset_path = raw_data_dir / "WA_Fn-UseC_-Telco-Customer-Churn-2.csv"
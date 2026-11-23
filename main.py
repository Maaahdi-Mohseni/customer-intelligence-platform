from src.data.load_data import load_data

df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn-2.csv")

print(df.head())
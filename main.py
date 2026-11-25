"""
This module serves as the entry point of the application.
"""
from sklearn.model_selection import train_test_split
from src.data.load_data import load_data
from src.models.train import train_model

df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn-2.csv")

X = df.drop(columns=(["Churn","customerID"]))
y = df["Churn"]

x_train , x_test , y_train , y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Train Model
model = train_model(x_train,y_train)

print("Model training completed")
print(type(model))
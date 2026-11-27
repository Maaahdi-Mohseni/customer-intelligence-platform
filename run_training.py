"""
This module serves as the entry point of the application.
"""
from sklearn.model_selection import train_test_split
from src.data.load_data import load_data
from src.models.train import train_model
from src.models.tune import tune_model
from src.models.evaluate import evaluate_model
from src.models.persistence import save_model

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

#Model pipeline
model_pipeline = train_model(x_train,y_train)

#Hyperparameter tuning
grid_search = tune_model(model_pipeline , x_train , y_train)

#Get best Model
best_model = grid_search.best_estimator_

#Evaluate best model
metrics = evaluate_model(best_model,x_test,y_test)

#Save model
save_model(best_model, "artifacts/churn_pipeline.joblib")

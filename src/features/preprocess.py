"""
This module handles data preprocessing.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def build_preprocessor():
    
    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]
    
    numerical_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen"
    ]
    
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer" , SimpleImputer(strategy="constant" , fill_value=0)),
            ("scaler" , StandardScaler())
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("Catgorical" , OneHotEncoder(handle_unknown="ignore") , categorical_features),
            ("Numerical", numerical_pipeline , numerical_features)
        ]
    )
    
    return preprocessor
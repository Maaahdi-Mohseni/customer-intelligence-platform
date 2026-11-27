"""
This module handles data preprocessing.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.features.schema import (
    categorical_features,
    numerical_features
)

def build_preprocessor():
    
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
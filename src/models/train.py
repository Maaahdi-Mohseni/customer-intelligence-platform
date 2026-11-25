"""
This module trains the model using data.
"""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.features.preprocess import build_preprocessor

def train_model(x_train,y_train):
    model_pipeline = Pipeline(steps=[
        ("Preprocessor" , build_preprocessor()),
        ("classifier" , LogisticRegression(C=10 , max_iter=1000))
    ])
    model_pipeline.fit(x_train,y_train)
    
    return model_pipeline
"""
This module saves and loads the model.
"""
from pathlib import Path 
import joblib

def save_model(model , file_path):
    file_path=Path(file_path)
    file_path.parent.mkdir(parents=True , exist_ok=True)
    joblib.dump(model , file_path)
    
def load_model(file_path):
    
    return joblib.load(file_path)
"""
This module evaluates the model.
"""
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    roc_auc_score
)

def evaluate_model(model,x_test,y_test):
    
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:,1]
    
    metrics = {
        "accuracy" : accuracy_score(y_test,y_pred),
        "precision" : precision_score(y_test,y_pred,pos_label="Yes"),
        "recall" : recall_score(y_test,y_pred,pos_label="Yes"),
        "f1" : f1_score(y_test,y_pred,pos_label="Yes"),
        "roc_auc" : roc_auc_score(
            (y_test=="Yes").astype(int),
            y_proba
        )
    }
    
    return metrics
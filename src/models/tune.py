"""
This module is used for model tuning.
"""
from sklearn.model_selection import GridSearchCV , StratifiedKFold
from sklearn.metrics import make_scorer , f1_score

def tune_model(model_pipline , x_train , y_train):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    f1_scorer = make_scorer(
        f1_score,
        pos_label ="Yes"
    )
    
    param_grid = {
        "Classifier" : [0.01, 0.1, 1, 10, 100]
    }
    
    grid_search = GridSearchCV(
        estimator=model_pipline,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=cv,
        n_jobs=-1
    )
    
    grid_search.fit(x_train,y_train)
    
    return grid_search
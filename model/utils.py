# Import all needed libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  classification_report, fbeta_score

# Mandatory functions for the project
def prepared_data (dataset: pd.DataFrame, target: str):
    x = dataset.drop(columns=[target])
    y = dataset[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 22, stratify = y)
    return x_train, x_test, y_train, y_test

def threshold_layer (model, x, threshold):
    # Obtaining probabilities
    probs = model.predict_proba(x)[:, 1]
    # Predictions with custom threshold
    y_custom_preds = (probs >= threshold).astype(int)
    return y_custom_preds

def threshold_f2_optimizer(model, x, y, start=0.01, end=1.0, step=0.01, beta=2.0):
    # Obtain probability results on the x_train
    probs = model.predict_proba(x)[:, 1]
    best_threshold = 0.5
    best_f2 = 0
    # Research range
    thresholds = np.arange(start, end + step, step)
    for threshold in thresholds:
        # Try the threshold
        preds = (probs >= threshold).astype(int)
        current_f2 = fbeta_score(y, preds, beta=beta, pos_label=1, zero_division=0)
        if current_f2 > best_f2:
            best_f2 = current_f2
            best_threshold = threshold     
    return round(best_threshold, 3), round(best_f2, 4)
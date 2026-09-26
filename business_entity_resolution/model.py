import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from typing import Tuple, List, Dict

FEATURE_COLS = [
    "name_ratio", "name_partial_ratio", "name_token_sort_ratio", "name_token_set_ratio", "name_exact",
    "addr_ratio", "addr_partial_ratio", "addr_token_sort_ratio", "addr_token_set_ratio", "addr_exact",
    "country_match",
]

def prepare_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLS].values
    y = df["label"].values
    return X, y

def train_lgbm(df_train: pd.DataFrame) -> lgb.Booster:
    X, y = prepare_xy(df_train)
    train_data = lgb.Dataset(X, label=y)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(params, train_data, num_boost_round=500)
    return model

def compute_f05_per_s1(y_true: np.ndarray, y_pred: np.ndarray, s1_ids: np.ndarray) -> np.ndarray:
    """
    Compute F0.5 per source1 entity, then return array of per-entity scores.
    y_true, y_pred are binary vectors at pair level.
    """
    unique_s1 = np.unique(s1_ids)
    scores = []

    for sid in unique_s1:
        mask = s1_ids == sid
        yt = y_true[mask]
        yp = y_pred[mask]

        tp = ((yt == 1) & (yp == 1)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if prec == 0 and rec == 0:
            f05 = 0.0
        else:
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
        scores.append(f05)

    return np.array(scores)

def find_best_threshold(
    model: lgb.Booster,
    df_val: pd.DataFrame,
) -> float:
    X, y = prepare_xy(df_val)
    s1_ids = df_val["s1_id"].values
    probs = model.predict(X)

    thresholds = np.linspace(0.1, 0.9, 17)
    best_t = 0.5
    best_score = -1

    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        scores = compute_f05_per_s1(y, y_pred, s1_ids)
        avg = scores.mean()
        if avg > best_score:
            best_score = avg
            best_t = t

    return best_t
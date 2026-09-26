import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .blocking import generate_candidates
from .config import MODEL_PATH, TRAIN_GT
from .features import build_pair_features
from .io_utils import load_ground_truth, load_train_sources, parse_matched_ids


def create_training_labels(gt: pd.DataFrame) -> dict:
    """Return dict[s1_id] -> set of matched S2/S3 IDs."""
    mapping = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched = set(parse_matched_ids(row["matched_entity_ids"]))
        mapping[s1_id] = matched
    return mapping


def main_train():
    train_limit = int(os.getenv("BERE_TRAIN_LIMIT", "400"))
    print(f"Training with row limit: {train_limit}")

    s1, s2, s3 = load_train_sources(limit=train_limit)
    gt = load_ground_truth(TRAIN_GT, s1_ids=set(s1["entity_id"]))
    label_map = create_training_labels(gt)

    candidates = generate_candidates(s1, s2, s3)
    s23 = pd.concat([s2, s3], ignore_index=True)
    pairs = build_pair_features(s1, s23, candidates)

    def get_label(row):
        s1_id = row["s1_id"]
        s23_id = row["s23_id"]
        return int(s23_id in label_map.get(s1_id, set()))

    pairs["label"] = pairs.apply(get_label, axis=1)

    s1_ids = pairs["s1_id"].unique()
    train_s1, val_s1 = train_test_split(s1_ids, test_size=0.2, random_state=42)

    train_mask = pairs["s1_id"].isin(train_s1)
    val_mask = pairs["s1_id"].isin(val_s1)

    df_train = pairs[train_mask].reset_index(drop=True)
    df_val = pairs[val_mask].reset_index(drop=True)

    from .model import find_best_threshold, train_lgbm

    model = train_lgbm(df_train)
    best_t = find_best_threshold(model, df_val)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": best_t}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}, threshold={best_t:.3f}")


if __name__ == "__main__":
    main_train()
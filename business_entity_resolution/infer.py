import os

import joblib
import pandas as pd

from .blocking import generate_candidates
from .config import MODEL_PATH, OUTPUT_DIR
from .features import build_pair_features
from .io_utils import get_id_sets_test, load_test_sources
from .model import FEATURE_COLS


def main_infer():
    test_limit = int(os.getenv("BERE_TEST_LIMIT", "400"))
    print(f"Inference with row limit: {test_limit}")

    s1, s2, s3 = load_test_sources(limit=test_limit)
    s1_ids_test, s2_ids_test, s3_ids_test = get_id_sets_test(s1, s2, s3)
    candidates = generate_candidates(s1, s2, s3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cand_rows = []
    for s1_id in s1["entity_id"]:
        cands = [c for c in candidates.get(s1_id, []) if c in s2_ids_test or c in s3_ids_test]
        cand_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cands),
        })
    pd.DataFrame(cand_rows).to_csv(OUTPUT_DIR / "candidate_pairs.tsv", sep="\t", index=False)

    s23 = pd.concat([s2, s3], ignore_index=True)
    pairs = build_pair_features(s1, s23, candidates)

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    threshold = artifact["threshold"]

    X = pairs[FEATURE_COLS].values
    probs = model.predict(X)
    pairs["prob"] = probs
    pairs["pred"] = (probs >= threshold).astype(int)

    pred_map = {}
    for s1_id in s1["entity_id"]:
        sub = pairs[pairs["s1_id"] == s1_id]
        matched = sub[sub["pred"] == 1]["s23_id"].tolist()
        matched = [m for m in matched if m in s2_ids_test or m in s3_ids_test]
        pred_map[s1_id] = sorted(set(matched))

    res_rows = [{
        "source1_entity_id": s1_id,
        "matched_entity_ids": ",".join(pred_map[s1_id]),
    } for s1_id in s1["entity_id"]]
    pd.DataFrame(res_rows).to_csv(OUTPUT_DIR / "matching_results.tsv", sep="\t", index=False)

    print("Outputs written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main_infer()
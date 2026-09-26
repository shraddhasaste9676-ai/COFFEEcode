from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd

from .config import TEST_S1, TEST_S2, TEST_S3, TRAIN_S1, TRAIN_S2, TRAIN_S3


def load_source(path: Path, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    if limit is not None:
        df = df.head(limit)
    df["source"] = df["entity_id"].str.split("-", n=1).str[0]
    return df


def load_train_sources(limit: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s1 = load_source(TRAIN_S1, limit=limit)
    s2 = load_source(TRAIN_S2, limit=limit)
    s3 = load_source(TRAIN_S3, limit=limit)
    return s1, s2, s3


def load_test_sources(limit: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s1 = load_source(TEST_S1, limit=limit)
    s2 = load_source(TEST_S2, limit=limit)
    s3 = load_source(TEST_S3, limit=limit)
    return s1, s2, s3


def load_ground_truth(path: Path, s1_ids: Set[str] | None = None) -> pd.DataFrame:
    gt = pd.read_csv(path, sep="\t", dtype=str)
    if s1_ids is not None:
        gt = gt[gt["source1_entity_id"].isin(s1_ids)].copy()
    return gt


def parse_matched_ids(matched_str: str) -> List[str]:
    if pd.isna(matched_str) or matched_str.strip() == "":
        return []
    return [x.strip() for x in matched_str.split(",") if x.strip()]


def get_id_sets_test(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> Tuple[Set[str], Set[str], Set[str]]:
    return set(s1["entity_id"]), set(s2["entity_id"]), set(s3["entity_id"])
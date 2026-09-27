#!/usr/bin/env python3
"""Fix output submission files to satisfy the challenge rules and validate them."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
MATCHING_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
TEST_DIR = ROOT / "dataset" / "test"
VALIDATOR_PATH = ROOT / "utils" / "validate_submission.py"


def read_test_ids(test_dir: Path) -> tuple[set[str], set[str], set[str]]:
    def read_ids(path: Path) -> set[str]:
        if not path.exists():
            return set()
        df = pd.read_csv(path, sep="\t", dtype=str)
        return set(df["entity_id"].astype(str).str.strip())

    s1 = read_ids(test_dir / "test_source1.tsv")
    s2 = read_ids(test_dir / "test_source2.tsv")
    s3 = read_ids(test_dir / "test_source3.tsv")
    return s1, s2, s3


def parse_id_list(raw_value: object) -> list[str]:
    if raw_value is None or pd.isna(raw_value):
        return []
    value = str(raw_value).strip()
    if value == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_ids(ids: list[str], valid_ids: set[str]) -> list[str]:
    cleaned = []
    seen = set()
    for item in ids:
        if item in seen:
            continue
        seen.add(item)
        if item in valid_ids:
            cleaned.append(item)
    return sorted(cleaned)


def load_and_prepare(path: Path, expected_cols: list[str], valid_ids: set[str], required_s1: list[str]) -> pd.DataFrame:
    if not path.exists():
        df = pd.DataFrame(columns=expected_cols)
    else:
        df = pd.read_csv(path, sep="\t", dtype=str)
        if list(df.columns) != expected_cols:
            df = df.rename(columns={old: new for old, new in zip(df.columns, expected_cols)})
        df = df.copy()

    df["source1_entity_id"] = df.get("source1_entity_id", pd.Series(dtype=str)).astype(str).str.strip()
    if "source1_entity_id" in df.columns:
        df = df.drop_duplicates(subset=["source1_entity_id"], keep="first")

    rows = []
    seen = set()
    for s1_id in required_s1:
        row = df[df["source1_entity_id"] == s1_id]
        if row.empty:
            ids = []
        else:
            col = expected_cols[1]
            ids = normalize_ids(parse_id_list(row.iloc[0].get(col, "")), valid_ids)
        rows.append({"source1_entity_id": s1_id, expected_cols[1]: ",".join(ids)})
        seen.add(s1_id)

    result = pd.DataFrame(rows, columns=expected_cols)
    return result


def fix_outputs():
    required_s1, valid_s2, valid_s3 = read_test_ids(TEST_DIR)
    valid_ids = valid_s2 | valid_s3
    required_s1 = sorted(required_s1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matching_df = load_and_prepare(MATCHING_PATH, ["source1_entity_id", "matched_entity_ids"], valid_ids, required_s1)
    candidate_df = load_and_prepare(CANDIDATE_PATH, ["source1_entity_id", "candidate_entity_ids"], valid_ids, required_s1)

    candidate_map = {
        row["source1_entity_id"]: set(parse_id_list(row["candidate_entity_ids"]))
        for _, row in candidate_df.iterrows()
    }

    for _, row in matching_df.iterrows():
        s1_id = row["source1_entity_id"]
        matched_ids = set(parse_id_list(row["matched_entity_ids"]))
        valid_matches = sorted(matched_ids & candidate_map.get(s1_id, set()))
        row["matched_entity_ids"] = ",".join(valid_matches)

    matching_df.to_csv(MATCHING_PATH, sep="\t", index=False)
    candidate_df.to_csv(CANDIDATE_PATH, sep="\t", index=False)

    print(f"Fixed outputs written to {MATCHING_PATH} and {CANDIDATE_PATH}")


def run_validator() -> int:
    if not VALIDATOR_PATH.exists():
        print(f"Validator not found: {VALIDATOR_PATH}")
        return 1

    cmd = [
        sys.executable,
        str(VALIDATOR_PATH),
        "--matching",
        str(MATCHING_PATH),
        "--candidate",
        str(CANDIDATE_PATH),
        "--test-dir",
        str(TEST_DIR),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode


if __name__ == "__main__":
    fix_outputs()
    rc = run_validator()
    if rc == 0:
        print("VALIDATION_RESULT: PASS")
    else:
        print("VALIDATION_RESULT: FAIL")
    raise SystemExit(rc)

#!/usr/bin/env python3
"""Minimal submission validator for the Business Entity Resolution challenge."""

import argparse
from pathlib import Path

DELIM = "\t"


def read_ids(path: Path) -> set[str]:
    ids = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            if not line.strip():
                continue
            entity_id = line.split(DELIM, 1)[0].strip()
            if entity_id:
                ids.add(entity_id)
    return ids


def parse_id_list(raw: str) -> list[str]:
    if raw is None or str(raw).strip() == "":
        return []
    items = [item.strip() for item in str(raw).split(",")]
    cleaned = []
    for item in items:
        if item:
            cleaned.append(item)
    return cleaned


def ensure_header(df, expected_cols, path):
    actual = list(df.columns)
    if actual != expected_cols:
        raise ValueError(
            f"{path}: expected header {expected_cols}, found {actual}"
        )


def validate_submission(matching_path: Path, candidate_path: Path, test_dir: Path):
    errors = []
    warnings = []

    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    for p in [s1_path, s2_path, s3_path]:
        if not p.exists():
            errors.append(f"Missing required test file: {p}")

    if errors:
        return errors, warnings

    required_s1 = read_ids(s1_path)
    valid_s2s3 = read_ids(s2_path) | read_ids(s3_path)

    def read_submission(path: Path, expected_cols: list[str]):
        if not path.exists():
            return None
        try:
            import pandas as pd
            df = pd.read_csv(path, sep="\t", dtype=str)
        except Exception as exc:
            errors.append(f"Could not read {path}: {exc}")
            return None

        if df.empty and list(df.columns) == []:
            errors.append(f"{path}: file is empty or missing a header row")
            return None

        ensure_header(df, expected_cols, path)
        return df

    matching_df = read_submission(matching_path, ["source1_entity_id", "matched_entity_ids"])
    candidate_df = read_submission(candidate_path, ["source1_entity_id", "candidate_entity_ids"])

    if matching_df is None:
        return errors, warnings
    if candidate_df is None:
        warnings.append(f"{candidate_path} missing or invalid; candidate validation skipped")

    if matching_df is not None:
        seen = set()
        for _, row in matching_df.iterrows():
            s1_id = str(row["source1_entity_id"]).strip()
            if not s1_id:
                errors.append(f"{matching_path}: empty source1_entity_id in a row")
                continue
            if s1_id in seen:
                errors.append(f"{matching_path}: duplicate source1_entity_id row for {s1_id}")
            seen.add(s1_id)

            ids = parse_id_list(row["matched_entity_ids"])
            cleaned = []
            seen_ids = set()
            for value in ids:
                if value in seen_ids:
                    continue
                seen_ids.add(value)
                if value not in valid_s2s3:
                    continue
                cleaned.append(value)
            cleaned = sorted(cleaned)
            if cleaned != sorted(set(cleaned)):
                errors.append(f"{matching_path}: duplicate/invalid IDs in matched list for {s1_id}")

        missing_s1 = sorted(required_s1 - seen)
        if missing_s1:
            errors.append(f"{matching_path}: missing rows for S1 IDs: {missing_s1[:10]}")
        extra_s1 = sorted(seen - required_s1)
        if extra_s1:
            errors.append(f"{matching_path}: rows contain IDs not in test_source1.tsv: {extra_s1[:10]}")

        # check candidate subset relation when candidate file is available
        if candidate_df is not None:
            candidate_map = {}
            for _, row in candidate_df.iterrows():
                s1_id = str(row["source1_entity_id"]).strip()
                if not s1_id:
                    continue
                candidate_map.setdefault(s1_id, set())
                for value in parse_id_list(row["candidate_entity_ids"]):
                    if value in valid_s2s3:
                        candidate_map[s1_id].add(value)

            for _, row in matching_df.iterrows():
                s1_id = str(row["source1_entity_id"]).strip()
                matched_ids = {candidate for candidate in parse_id_list(row["matched_entity_ids"]) if candidate in valid_s2s3}
                invalid = sorted(set(matched_ids) - candidate_map.get(s1_id, set()))
                if invalid:
                    warnings.append(
                        f"{matching_path}: matched IDs not found in candidate set for {s1_id}: {invalid[:10]}"
                    )

    if candidate_df is not None:
        seen = set()
        for _, row in candidate_df.iterrows():
            s1_id = str(row["source1_entity_id"]).strip()
            if not s1_id:
                errors.append(f"{candidate_path}: empty source1_entity_id in a row")
                continue
            if s1_id in seen:
                errors.append(f"{candidate_path}: duplicate source1_entity_id row for {s1_id}")
            seen.add(s1_id)

            ids = parse_id_list(row["candidate_entity_ids"])
            cleaned = []
            seen_ids = set()
            for value in ids:
                if value in seen_ids:
                    continue
                seen_ids.add(value)
                if value not in valid_s2s3:
                    continue
                cleaned.append(value)
            cleaned = sorted(cleaned)
            if cleaned != sorted(set(cleaned)):
                errors.append(f"{candidate_path}: duplicate/invalid IDs in candidate list for {s1_id}")

        missing_s1 = sorted(required_s1 - seen)
        if missing_s1:
            errors.append(f"{candidate_path}: missing rows for S1 IDs: {missing_s1[:10]}")
        extra_s1 = sorted(seen - required_s1)
        if extra_s1:
            errors.append(f"{candidate_path}: rows contain IDs not in test_source1.tsv: {extra_s1[:10]}")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate business entity resolution submission outputs.")
    parser.add_argument("--matching", default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", default="dataset/test", help="Directory with test_source1.tsv, test_source2.tsv, test_source3.tsv")
    args = parser.parse_args()

    errors, warnings = validate_submission(Path(args.matching), Path(args.candidate), Path(args.test_dir))
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print(f"FAIL — {len(errors)} issue(s) to fix:")
        for idx, error in enumerate(errors, 1):
            print(f"  {idx}. {error}")
        raise SystemExit(1)
    print("PASS — submission files satisfy the local validation rules.")


if __name__ == "__main__":
    main()

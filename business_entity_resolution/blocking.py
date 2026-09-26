import re
from collections import defaultdict
from typing import Dict, List, Set

import pandas as pd

LEGAL_SUFFIXES = {
    "ltd", "limited", "pvt", "private", "corp", "corporation",
    "llc", "llp", "inc", "incorporated", "co", "company", "gmbh", "plc"
}

ABBREVS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "nr": "near",
    "opp": "opposite",
    "nd": "",
    "th": "",
}


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    t = str(text).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def remove_legal_suffixes(name: str) -> str:
    t = normalize_text(name)
    tokens = t.split()
    tokens = [tok for tok in tokens if tok not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def normalize_address(addr: str) -> str:
    t = normalize_text(addr)
    tokens = t.split()
    norm_tokens = []
    for tok in tokens:
        norm_tokens.append(ABBREVS.get(tok, tok))
    return " ".join(norm_tokens)


def name_prefix_tokens(name: str, k: int = 2) -> List[str]:
    t = remove_legal_suffixes(name)
    tokens = t.split()
    if not tokens:
        return []
    return ["_".join(tokens[: min(k, len(tokens))])]


def name_first_char_len_bucket(name: str) -> List[str]:
    t = remove_legal_suffixes(name)
    if not t:
        return []
    first = t[0]
    length_bucket = len(t) // 5
    return [f"{first}_{length_bucket}"]


def address_numeric_prefix(addr: str) -> List[str]:
    t = normalize_address(addr)
    nums = [tok for tok in t.split() if tok.isdigit()]
    return [nums[0]] if nums else []


def address_city_country(addr: str, country: str) -> List[str]:
    t = normalize_address(addr)
    tokens = t.split()
    if len(tokens) >= 3:
        city_part = "_".join(tokens[-3:-1])
    elif len(tokens) >= 2:
        city_part = "_".join(tokens[:2])
    else:
        city_part = t
    return [f"{country.lower()}_{city_part}"] if city_part else []


def compute_blocking_keys_row(row: pd.Series) -> List[str]:
    name = row.get("business_name", "")
    addr = row.get("business_address", "")
    country = row.get("country", "")
    country_key = str(country).lower()

    keys: List[str] = []

    for k in [2, 3]:
        for prefix in name_prefix_tokens(name, k=k):
            keys.append(f"name_prefix{k}_{prefix}_{country_key}")

    for key in name_first_char_len_bucket(name):
        keys.append(f"name_charlen_{key}_{country_key}")

    for num in address_numeric_prefix(addr):
        keys.append(f"addr_num_{num}_{country_key}")

    for city_key in address_city_country(addr, country):
        keys.append(f"addr_city_{city_key}")

    return keys


def build_blocking_index(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return a blocking key -> list of entity IDs mapping."""
    index: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        entity_id = row["entity_id"]
        for key in compute_blocking_keys_row(row):
            index[key].append(entity_id)
    return index


def generate_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> Dict[str, List[str]]:
    """Return a candidate list for each S1 entity against S2 and S3."""
    s23 = pd.concat([s2, s3], ignore_index=True)
    block_index = build_blocking_index(s23)

    candidates: Dict[str, List[str]] = {}
    for _, row in s1.iterrows():
        cand_set: Set[str] = set()
        for key in compute_blocking_keys_row(row):
            cand_set.update(block_index.get(key, []))
        candidates[row["entity_id"]] = sorted(cand_set)

    return candidates

import re
from typing import Dict, List, Set

import pandas as pd
from rapidfuzz import fuzz

from .blocking import normalize_address, remove_legal_suffixes

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


def name_similarity_features(name1: str, name2: str) -> dict:
    n1 = remove_legal_suffixes(name1)
    n2 = remove_legal_suffixes(name2)

    return {
        "name_ratio": fuzz.ratio(n1, n2) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(n1, n2) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(n1, n2) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(n1, n2) / 100.0,
        "name_exact": float(n1 == n2 and n1 != ""),
        "name_len_diff": abs(len(n1) - len(n2)),
        "name_token_overlap": len(set(n1.split()) & set(n2.split())),
    }


def address_similarity_features(addr1: str, addr2: str) -> dict:
    a1 = normalize_address(addr1)
    a2 = normalize_address(addr2)

    return {
        "addr_ratio": fuzz.ratio(a1, a2) / 100.0,
        "addr_partial_ratio": fuzz.partial_ratio(a1, a2) / 100.0,
        "addr_token_sort_ratio": fuzz.token_sort_ratio(a1, a2) / 100.0,
        "addr_token_set_ratio": fuzz.token_set_ratio(a1, a2) / 100.0,
        "addr_exact": float(a1 == a2 and a1 != ""),
        "addr_len_diff": abs(len(a1) - len(a2)),
        "addr_token_overlap": len(set(a1.split()) & set(a2.split())),
    }


def extract_numeric_prefix(addr: str) -> str:
    a = normalize_address(addr)
    tokens = a.split()
    for tok in tokens:
        if tok.isdigit():
            return tok
    return ""


def cross_features(row1: pd.Series, row2: pd.Series) -> dict:
    feats = {}
    feats["country_match"] = float(row1["country"] == row2["country"])

    num1 = extract_numeric_prefix(row1["business_address"])
    num2 = extract_numeric_prefix(row2["business_address"])
    feats["addr_num_exact"] = float(num1 == num2 and num1 != "")
    if num1.isdigit() and num2.isdigit():
        feats["addr_num_diff"] = abs(int(num1) - int(num2))
    else:
        feats["addr_num_diff"] = 999

    return feats


def build_pair_features(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    candidate_dict: Dict[str, List[str]],
    add_label: bool = False,
    label_map: Dict[str, Set[str]] | None = None,
) -> pd.DataFrame:
    """Build pair-level feature dataframe for Source 1 vs. Source 2/3 candidates."""
    if s1.empty or s23.empty:
        return pd.DataFrame(columns=["s1_id", "s23_id"])

    s1_index = s1.set_index("entity_id")
    s23_index = s23.set_index("entity_id")
    rows: List[dict] = []

    for s1_id, cands in candidate_dict.items():
        if s1_id not in s1_index.index:
            continue
        row1 = s1_index.loc[s1_id]

        for s23_id in cands:
            if s23_id not in s23_index.index:
                continue
            row2 = s23_index.loc[s23_id]

            feats: Dict[str, float | int | str] = {
                "s1_id": s1_id,
                "s23_id": s23_id,
            }
            feats.update(name_similarity_features(row1["business_name"], row2["business_name"]))
            feats.update(address_similarity_features(row1["business_address"], row2["business_address"]))
            feats.update(cross_features(row1, row2))

            if add_label and label_map is not None:
                feats["label"] = int(s23_id in label_map.get(s1_id, set()))

            rows.append(feats)

    return pd.DataFrame(rows)

import json
import pandas as pd


def load_manual_corrections(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["chunk_id", "final_label"])
    return pd.read_json(path)


def merge_weak_and_manual(weak_df: pd.DataFrame, manual_df: pd.DataFrame):
    """Fusionne les corrections manuelles dans le dataset weak."""
    merged = weak_df.merge(
        manual_df,
        on="chunk_id",
        how="left",
        suffixes=("_weak", "_manual")
    )

    # si correction dispo, on remplace
    merged["final_label"] = merged.apply(
        lambda row: row["final_label_manual"]
        if pd.notna(row["final_label_manual"])
        else row["final_label_weak"],
        axis=1
    )

    return merged.drop(columns=["final_label_manual"])

# src/models/utils.py
"""
Petits utilitaires : id generation, text cleaning, saving helpers.
"""

import uuid
import re
from typing import List
import json
import os
import numpy as np

def make_chunk_id(report_id: str, start: int) -> str:
    return f"{report_id}_{start}"

def generate_uuid(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex

def simple_clean_text(text: str) -> str:
    # remove non printable and normalize spaces
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_numpy(path: str):
    return np.load(path, allow_pickle=True)


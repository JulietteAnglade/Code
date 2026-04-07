#!/usr/bin/env python3
"""Compute top-20 cosine similarities between articles and 'narratifs orth'.

Usage:
    python compute_top20_with_narratifs_orth.py \
        --narratives data_narratives/narratives_orth.csv \
        --articles data_narratives/resampled_articles_with_embeddings.json \
        --out results_narratives/top20_per_narrative.json \
        --out-global results_narratives/top20_global.json

The script handles large JSON files by attempting a JSONL (one JSON object per line)
read first, then falling back to loading as a JSON array if needed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd


def parse_embedding_str(s: str) -> np.ndarray:
    if pd.isna(s):
        return np.array([])
    # Flatten multiline bracketed numpy repr into numbers
    # Accept scientific notation
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", s)
    return np.array([float(x) for x in nums], dtype=float)


def load_narratives(narr_path: str) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(narr_path, dtype=str)
    if "embedding" not in df.columns:
        raise ValueError("narratives CSV must have an 'embedding' column")
    embeddings = [parse_embedding_str(s) for s in df["embedding"].tolist()]
    # Check consistent dims
    dims = {e.size for e in embeddings}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent narrative embedding sizes: {dims}")
    mat = np.vstack(embeddings)
    titles = df.get("title", pd.Series([None] * len(df))).tolist()
    return titles, mat


def try_load_jsonl(path: str):
    objs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError:
                return None
    return objs


def load_articles_embeddings(path: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    # Try JSONL first
    objs = try_load_jsonl(path)
    if objs is None:
        # fallback: try load whole file as JSON
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            objs = data
        elif isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
            objs = data["items"]
        else:
            raise ValueError("Unrecognized JSON format for articles embeddings")

    embeddings = []
    records = []
    for idx, o in enumerate(objs):
        emb = None
        if isinstance(o, dict):
            # common keys might be 'embedding' or 'embeddings'
            if "embedding" in o:
                emb = o["embedding"]
            elif "embeddings" in o:
                emb = o["embeddings"]
        # If emb is string, parse
        if isinstance(emb, str):
            vec = parse_embedding_str(emb)
        elif isinstance(emb, (list, tuple)):
            vec = np.array(emb, dtype=float)
        else:
            # try to find numeric list anywhere in object
            # fallback: search values
            found = None
            for v in o.values() if isinstance(o, dict) else []:
                if isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
                    found = v
                    break
            if found is not None:
                vec = np.array(found, dtype=float)
            else:
                raise ValueError(f"No embedding found in article record at index {idx}")

        embeddings.append(vec)
        records.append({"index": idx, **({k: o.get(k) for k in ("title", "id", "url") if isinstance(o, dict)})})

    dims = {e.size for e in embeddings}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent article embedding sizes: {dims}")
    mat = np.vstack(embeddings)
    return records, mat


def topk_similarities(query_vec: np.ndarray, corpus: np.ndarray, k: int = 20) -> List[Tuple[int, float]]:
    # Normalize to compute cosine similarity via dot
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return []
    q = query_vec / q_norm
    norms = np.linalg.norm(corpus, axis=1)
    # Avoid divide by zero
    valid = norms > 0
    if not np.any(valid):
        return []
    corpus_normed = corpus[valid] / norms[valid][:, None]
    sims = corpus_normed.dot(q)
    # get top-k indices in the original corpus
    if sims.size == 0:
        return []
    k = min(k, sims.size)
    idx_part = np.argpartition(-sims, k - 1)[:k]
    top_idx_sorted = idx_part[np.argsort(-sims[idx_part])]
    # map back to original indices
    orig_indices = np.nonzero(valid)[0][top_idx_sorted]
    return [(int(i), float(sims[top_idx_sorted][j])) for j, i in enumerate(orig_indices)]


def compute_top20(narr_titles: List[str], narr_mat: np.ndarray, art_records: List[Dict[str, Any]], art_mat: np.ndarray):
    per_narr = {}
    for i, title in enumerate(narr_titles):
        q = narr_mat[i]
        top = topk_similarities(q, art_mat, k=20)
        per_narr[title or f"narr_{i}"] = [{"article_index": idx, "similarity": sim} for idx, sim in top]

    # global top-20 pairs
    # Compute similarity matrix in blocks to avoid huge memory if needed
    # For simplicity compute full matrix when sizes small; otherwise compute per narrative and aggregate
    pairs = []
    for i in range(narr_mat.shape[0]):
        q = narr_mat[i]
        top = topk_similarities(q, art_mat, k=20)
        for idx, sim in top:
            pairs.append({"narrative_index": i, "article_index": idx, "similarity": sim})
    # keep top 20 globally
    pairs_sorted = sorted(pairs, key=lambda x: -x["similarity"])[:20]
    return per_narr, pairs_sorted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--narratives", default="data_narratives/narratives_orth.csv")
    p.add_argument("--articles", default="data_narratives/resampled_articles_with_embeddings.json")
    p.add_argument("--out", default="results_narratives/top20_per_narrative.json")
    p.add_argument("--out-global", default="results_narratives/top20_global.json")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading narratives from", args.narratives)
    titles, narr_mat = load_narratives(args.narratives)
    print("Narratives loaded:", narr_mat.shape)

    print("Loading article embeddings from", args.articles)
    art_records, art_mat = load_articles_embeddings(args.articles)
    print("Articles loaded:", art_mat.shape)

    if narr_mat.shape[1] != art_mat.shape[1]:
        raise ValueError(f"Embedding dimension mismatch: narratives {narr_mat.shape[1]} vs articles {art_mat.shape[1]}")

    per_narr, global_top = compute_top20(titles, narr_mat, art_records, art_mat)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(per_narr, f, ensure_ascii=False, indent=2)
    with open(args.out_global, "w", encoding="utf-8") as f:
        json.dump(global_top, f, ensure_ascii=False, indent=2)

    print("Wrote per-narrative top20 to", args.out)
    print("Wrote global top20 to", args.out_global)


if __name__ == "__main__":
    main()

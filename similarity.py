#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Similarity, deduplication, and clustering helpers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from core import DOMAINS
from models import TopicModel


def parse_existing_titles_from_csv(df: pd.DataFrame) -> List[str]:
    cols_norm = [c.lower().strip() for c in df.columns]
    title_col = None
    for candidate in ["title", "topic", "topic_title", "name", "question", "question_title"]:
        if candidate in cols_norm:
            title_col = df.columns[cols_norm.index(candidate)]
            break
    if title_col is None:
        for c in df.columns:
            if df[c].dtype == object:
                title_col = c
                break
    if title_col is None:
        return []

    titles = df[title_col].astype(str).map(lambda s: s.strip()).tolist()
    titles = [t for t in titles if t and t.lower() != "nan"]

    seen = set()
    out = []
    for t in titles:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def parse_existing_titles_from_text(raw: str) -> List[str]:
    lines = [ln.strip() for ln in (raw or "").splitlines()]
    lines = [re.sub(r"^\s*[-*•]\s+", "", ln).strip() for ln in lines]
    lines = [ln for ln in lines if ln]
    seen = set()
    out = []
    for t in lines:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def build_vectorizer_and_matrix(texts: List[str]) -> Tuple[TfidfVectorizer, Any]:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    X = vec.fit_transform(texts)
    return vec, X


def cosine_sim_matrix(A, B) -> np.ndarray:
    return (A @ B.T).toarray()


def annotate_similarity_to_existing(cands: List[TopicModel], existing_titles: List[str]) -> None:
    if not cands:
        return
    if not existing_titles:
        for t in cands:
            t.max_sim_existing = 0.0
            t.nearest_existing = ""
        return

    cand_texts = [f"{t.title} {t.summary}" for t in cands]
    _, X = build_vectorizer_and_matrix(existing_titles + cand_texts)
    X_exist = X[: len(existing_titles)]
    X_cand = X[len(existing_titles) :]

    sims = cosine_sim_matrix(X_cand, X_exist)
    for i, t in enumerate(cands):
        j = int(np.argmax(sims[i]))
        t.max_sim_existing = float(sims[i, j])
        t.nearest_existing = existing_titles[j]


def base_score(t: TopicModel) -> float:
    return (1.0 - float(t.max_sim_existing)) + 0.05 * min(len(t.summary) / 180.0, 1.0)


def dedup_candidates(
    cands: List[TopicModel],
    existing_titles: List[str],
    th_existing: float,
    th_within: float,
) -> List[TopicModel]:
    if not cands:
        return []
    annotate_similarity_to_existing(cands, existing_titles)
    cands = [t for t in cands if float(t.max_sim_existing) < float(th_existing)]
    if len(cands) <= 1:
        return cands

    texts = [f"{t.title} {t.summary}" for t in cands]
    _, X = build_vectorizer_and_matrix(texts)
    sims = cosine_sim_matrix(X, X)

    order = list(np.argsort(-np.array([base_score(t) for t in cands])))

    kept_idx: List[int] = []
    for i in order:
        if not kept_idx:
            kept_idx.append(int(i))
            continue
        max_sim = max(float(sims[i, j]) for j in kept_idx)
        if max_sim < float(th_within):
            kept_idx.append(int(i))

    kept = [cands[i] for i in kept_idx]
    annotate_similarity_to_existing(kept, existing_titles)
    return kept


def assign_clusters(
    cands: List[TopicModel],
    n_clusters: int,
    random_state: int = 42,
) -> None:
    if not cands:
        return
    n_clusters = int(max(2, min(n_clusters, len(cands))))
    texts = [f"{t.title} {t.summary}" for t in cands]
    vec = TfidfVectorizer(stop_words="english", max_features=4000)
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = km.fit_predict(X)
    for t, lab in zip(cands, labels):
        t.cluster_id = int(lab)


def cluster_balanced_select(
    cands: List[TopicModel],
    k: int,
    max_per_domain: int,
) -> List[TopicModel]:
    if not cands or k <= 0:
        return []
    k = min(int(k), len(cands))

    clusters: Dict[int, List[TopicModel]] = {}
    for t in cands:
        clusters.setdefault(int(t.cluster_id), []).append(t)
    for cid in clusters:
        clusters[cid].sort(key=base_score, reverse=True)

    domain_counts: Dict[str, int] = {d: 0 for d in DOMAINS}
    picked: List[TopicModel] = []

    cluster_order = sorted(clusters.keys(), key=lambda c: (-len(clusters[c]), c))

    while len(picked) < k:
        progressed = False
        for cid in cluster_order:
            if len(picked) >= k:
                break
            bucket = clusters.get(cid, [])
            while bucket:
                cand = bucket.pop(0)
                if domain_counts.get(cand.domain, 0) >= int(max_per_domain):
                    continue
                picked.append(cand)
                domain_counts[cand.domain] = domain_counts.get(cand.domain, 0) + 1
                progressed = True
                break
        if not progressed:
            break

    return picked


def default_cluster_count(n: int) -> int:
    return max(3, min(12, int(round(n**0.5))))

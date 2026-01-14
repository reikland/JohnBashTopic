#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metaculus Topic Generator — OpenRouter only — Faker-seeded + Clustering (minimal, robust)

Pipeline:
1) Faker seed-pack (themes; forbidden city/company tokens)
2) GEN (plain-text blocks with strict delimiters) -> blocks
3) FORMAT (per-block, temp=0) -> strict JSON Topic objects (Pydantic-validated)
4) (optional) JUDGE (per-topic) -> accept / revise / reject
5) Dedup vs existing + within-batch (TF-IDF char-ngrams cosine)
6) CLUSTER (KMeans over TF-IDF vectors) + cluster-balanced selection
7) Export CSV + JSON

Install:
  pip install streamlit requests pandas scikit-learn numpy faker pydantic

Run:
  streamlit run topicgen_faker_cluster.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from faker import Faker
from pydantic import BaseModel, Field, ValidationError, field_validator
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# Constants
# =========================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT_S = 60

DOMAINS = [
    "Geopolitics & Security",
    "Macroeconomics",
    "Markets & Finance",
    "AI & Compute",
    "Science & Biotech",
    "Energy & Climate",
    "Law & Regulation",
    "Technology & Industry",
    "Public Health",
    "Space",
    "Sports",
    "Culture & Media",
]

DEFAULT_GEN_MODEL = "openai/gpt-4o-mini"
DEFAULT_FORMATTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"

# Anti-repeat thresholds (tune)
DEFAULT_TH_EXISTING = 0.78   # candidate vs existing titles
DEFAULT_TH_WITHIN = 0.86     # candidate vs candidate

# Prompt size guardrails
MAX_EXISTING_TITLES_IN_PROMPT = 200
MAX_EXISTING_CHARS_IN_PROMPT = 12000


# =========================
# Small helpers
# =========================

def _sleep_backoff(attempt: int, base_s: float = 0.6, cap_s: float = 6.0) -> None:
    jitter = random.random() * 0.25
    t = min(cap_s, base_s * (2 ** (attempt - 1))) + jitter
    time.sleep(t)

def _clean_line(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\s*[\-\*\u2022]\s+", "", s)
    s = re.sub(r"^\s*\d+\s*[\.\)]\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_first_balanced_json(text: str) -> str:
    """
    Extract the first balanced JSON object/array from a string (robust against pre/post text).
    """
    if not text:
        raise ValueError("Empty text")

    i_obj = text.find("{")
    i_arr = text.find("[")
    if i_obj == -1 and i_arr == -1:
        raise ValueError("No JSON start found")

    if i_obj == -1:
        start, open_ch, close_ch = i_arr, "[", "]"
    elif i_arr == -1:
        start, open_ch, close_ch = i_obj, "{", "}"
    else:
        start, open_ch, close_ch = (i_arr, "[", "]") if i_arr < i_obj else (i_obj, "{", "}")

    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : j + 1]
    raise ValueError("No balanced JSON found")

def safe_json_loads(text: str) -> Any:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    return json.loads(_extract_first_balanced_json(text))

def pack_existing_for_prompt(existing_titles: List[str]) -> str:
    titles = existing_titles[:MAX_EXISTING_TITLES_IN_PROMPT]
    blob = "\n".join(f"- {t}" for t in titles)
    if len(blob) > MAX_EXISTING_CHARS_IN_PROMPT:
        blob = blob[:MAX_EXISTING_CHARS_IN_PROMPT] + "\n- (truncated)"
    return blob


# =========================
# OpenRouter client
# =========================

def openrouter_chat(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: float = REQUEST_TIMEOUT_S,
    retries: int = 3,
) -> str:
    if not (api_key or "").strip():
        raise RuntimeError("Missing OpenRouter API key.")

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    # Optional recommended headers
    ref = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_APP_TITLE")
    if ref:
        headers["HTTP-Referer"] = ref
    if title:
        headers["X-Title"] = title

    payload = {
        "model": model.strip(),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "messages": messages,
    }

    last_err: Optional[BaseException] = None
    last_text = ""

    for attempt in range(1, retries + 1):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_s)
            last_text = r.text
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < retries:
                _sleep_backoff(attempt)
                continue

    raise RuntimeError(
        "OpenRouter request failed.\n"
        f"Last error: {repr(last_err)}\n"
        f"Last response text (truncated): {last_text[:700]}"
    )


# =========================
# Faker seeding (guarded)
# =========================

@dataclass
class SeedPack:
    job: str
    industry: str
    country: str
    buzzwords: List[str]
    event: str
    forbidden_city: str
    forbidden_company: str

def make_faker(locale: str, seed: Optional[int]) -> Faker:
    fk = Faker(locale)
    if seed is not None:
        Faker.seed(seed)
        random.seed(seed)
    return fk

def sample_seed_pack(fk: Faker) -> SeedPack:
    buzz_pool = [
        "regulation", "antitrust", "supply chain", "interest rates", "inflation",
        "cybersecurity", "AI safety", "semiconductors", "energy transition",
        "labor market", "mergers", "IPO", "venture capital", "sanctions",
        "defense procurement", "biotech", "clinical trial", "data privacy",
    ]
    industry_pool = [
        "banking", "insurance", "healthcare", "pharmaceuticals", "energy",
        "telecommunications", "aviation", "logistics", "retail", "manufacturing",
        "software", "semiconductors", "automotive", "real estate",
    ]
    event_pool = [
        "earnings call", "regulatory investigation", "product recall", "data breach",
        "strike", "major acquisition", "rate decision", "trade dispute",
        "export restrictions", "bank stress",
    ]

    return SeedPack(
        job=fk.job(),
        industry=random.choice(industry_pool),
        country=fk.country(),
        buzzwords=random.sample(buzz_pool, k=4),
        event=random.choice(event_pool),
        forbidden_city=fk.city(),
        forbidden_company=fk.company(),
    )

def forbidden_tokens(seed: SeedPack) -> List[str]:
    toks = []
    for s in [seed.forbidden_city, seed.forbidden_company]:
        s = (s or "").strip()
        if s and len(s) >= 4:
            toks.append(s.lower())
    return toks

def contains_forbidden(text: str, forbidden: List[str]) -> bool:
    low = (text or "").lower()
    return any(tok in low for tok in forbidden if tok)


# =========================
# Models (Pydantic)
# =========================

class TopicModel(BaseModel):
    title: str = Field(min_length=6, max_length=220)
    summary: str = Field(default="—", max_length=900)
    domain: str
    key_entities: List[str] = Field(default_factory=list, max_length=10)
    question_hooks: List[str] = Field(default_factory=list, max_length=5)
    resolvability_note: str = Field(default="Resolvable via reputable public sources.", max_length=500)
    novelty_note: str = Field(default="", max_length=500)

    # Enrichment fields (not required from formatter)
    max_sim_existing: float = 0.0
    nearest_existing: str = ""
    judge_verdict: str = ""
    judge_rationale: str = ""
    cluster_id: int = -1

    @field_validator("title")
    @classmethod
    def _clean_title(cls, v: str) -> str:
        v = _clean_line(v)
        if "?" in v:
            raise ValueError("Title must be a topic title, not a question.")
        return v

    @field_validator("domain")
    @classmethod
    def _valid_domain(cls, v: str) -> str:
        v = (v or "").strip()
        if v not in DOMAINS:
            raise ValueError(f"Invalid domain: {v!r}")
        return v

    @field_validator("key_entities", "question_hooks")
    @classmethod
    def _clean_list(cls, v: List[str]) -> List[str]:
        out = []
        for x in (v or []):
            s = _clean_line(str(x))
            if s:
                out.append(s)
        return out


# =========================
# Existing titles ingestion
# =========================

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


# =========================
# Generation: plain-text blocks
# =========================

def build_generation_messages(
    n: int,
    resolve_by_iso: str,
    open_time: dt.datetime,
    existing_titles: List[str],
    allowed_domains: List[str],
    seed: SeedPack,
) -> List[Dict[str, str]]:
    domain_str = ", ".join(allowed_domains)
    existing_blob = pack_existing_for_prompt(existing_titles) if existing_titles else ""
    forbidden = f'FORBIDDEN (often fictional): city="{seed.forbidden_city}", company="{seed.forbidden_company}"'

    system = (
        "You generate high-quality, diverse forecasting TOPIC IDEAS suitable for Metaculus.\n"
        "Output MUST be plain text with strict delimiters (NOT JSON).\n"
        "Be specific, time-bounded, and publicly resolvable.\n"
        "Avoid vague words like 'trend', 'outlook', 'analysis', 'developments', 'impact'.\n"
    )

    user = (
        f"OPEN_TIME (now): {open_time.isoformat(timespec='seconds')}\n"
        f"Generate exactly {n} topic ideas resolvable by {resolve_by_iso}.\n"
        "IMPORTANT (anti-past): Do NOT generate topics whose outcomes are already known strictly before OPEN_TIME.\n"
        "Use a future-facing time anchor in the title (month+year preferred, e.g., 'by March 2026', 'in 2026').\n\n"
        "SEED PACK (for thematic diversity; do NOT copy forbidden strings):\n"
        f"- Job theme: {seed.job}\n"
        f"- Industry theme: {seed.industry}\n"
        f"- Country anchor (use this or another real anchor): {seed.country}\n"
        f"- Event type: {seed.event}\n"
        f"- Buzzwords: {', '.join(seed.buzzwords)}\n"
        f"- {forbidden}\n"
        "=> Never include the forbidden city/company strings.\n\n"
        "Return each topic in this exact template, then a line with ONLY '---' as a separator.\n\n"
        "TEMPLATE (MUST follow exactly):\n"
        "TITLE: <<=18 words>\n"
        f"DOMAIN: <one of: {domain_str}>\n"
        "SUMMARY: <2-4 sentences>\n"
        "KEY_ENTITIES: <comma-separated, 0-6 items>\n"
        "QUESTION_HOOKS: <2-3 hooks separated by ' || '>\n"
        "RESOLVABILITY: <public sources likely>\n"
        "NOVELTY: <why not a repeat>\n"
        "---\n\n"
        "Diversity requirements:\n"
        "- Spread across multiple domains.\n"
        "- Avoid repeating the same countries/companies across many topics.\n"
        "- Use real entities only (real regulators/orgs/countries/companies/indices).\n"
    )

    if existing_blob:
        user += "\nExisting topics to avoid (titles):\n" + existing_blob + "\n"

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def split_topic_blocks(raw_text: str) -> List[str]:
    t = (raw_text or "").replace("\r\n", "\n").strip()
    parts: List[str] = []
    buf: List[str] = []
    for ln in t.split("\n"):
        if ln.strip() == "---":
            block = "\n".join(buf).strip()
            if block:
                parts.append(block)
            buf = []
        else:
            buf.append(ln)
    tail = "\n".join(buf).strip()
    if tail:
        parts.append(tail)

    parts = [p for p in parts if "TITLE:" in p and "DOMAIN:" in p and "SUMMARY:" in p]
    return parts


# =========================
# Formatter: per-block -> strict JSON topic
# =========================

def format_block_to_topic(
    api_key: str,
    model: str,
    block: str,
    resolve_by_iso: str,
    open_time: dt.datetime,
    allowed_domains: List[str],
    retries: int = 3,
) -> TopicModel:
    domain_str = ", ".join(allowed_domains)

    system = (
        "You are a strict JSON formatter for Metaculus topic objects.\n"
        "Return ONLY valid JSON (one object). No markdown. No extra text.\n"
        "Domain MUST be exactly one of the allowed domains.\n"
        "Keep the TITLE as a topic title (not a question).\n"
    )

    user = (
        f"OPEN_TIME (now): {open_time.isoformat(timespec='seconds')}\n"
        f"Deadline: {resolve_by_iso}\n"
        f"Allowed domains: {domain_str}\n\n"
        "Convert this block into a JSON object with EXACT keys:\n"
        "title, summary, domain, key_entities, question_hooks, resolvability_note, novelty_note\n\n"
        "BLOCK:\n"
        + block
    )

    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_err: Optional[BaseException] = None

    for attempt in range(1, retries + 1):
        try:
            out = openrouter_chat(api_key, model, msgs, temperature=0.0, max_tokens=900, retries=2)
            obj = safe_json_loads(out)
            if not isinstance(obj, dict):
                raise ValueError("Formatter did not return a JSON object.")
            if isinstance(obj.get("question_hooks"), str):
                obj["question_hooks"] = [p.strip() for p in obj["question_hooks"].split("||") if p.strip()]
            if isinstance(obj.get("key_entities"), str):
                obj["key_entities"] = [p.strip() for p in obj["key_entities"].split(",") if p.strip()]

            try:
                return TopicModel(**obj)
            except ValidationError as ve:
                raise ValueError(str(ve)) from ve
        except Exception as e:
            last_err = e
            if attempt < retries:
                msgs.append(
                    {"role": "user",
                     "content": "Your previous output was invalid JSON or invalid schema. Return ONLY a valid JSON object with the required keys."}
                )
                _sleep_backoff(attempt)

    raise RuntimeError(f"Formatter failed after {retries} retries: {last_err}")


# =========================
# Optional judge: per-topic -> accept / revise / reject
# =========================

def judge_topic(
    api_key: str,
    model: str,
    topic: TopicModel,
    resolve_by_iso: str,
    open_time: dt.datetime,
    sim_existing: float,
    nearest_existing: str,
    allowed_domains: List[str],
    retries: int = 2,
) -> Optional[TopicModel]:
    domain_str = ", ".join(allowed_domains)

    system = (
        "You are a strict Metaculus editor.\n"
        "Return ONLY valid JSON (one object). No markdown, no extra text.\n"
        "Decide accept/reject/revise.\n"
        "Reject near-duplicates or unresolvable/vague topics.\n"
        "If already resolved/known strictly before OPEN_TIME, verdict MUST be reject.\n"
        "If revise: make it clearly distinct and more precise, still resolvable by the deadline.\n"
    )

    user = (
        f"OPEN_TIME (now): {open_time.isoformat(timespec='seconds')}\n"
        f"Deadline: {resolve_by_iso}\n"
        f"Allowed domains: {domain_str}\n"
        f"Nearest existing title: {nearest_existing}\n"
        f"Similarity to nearest existing (0-1): {sim_existing:.3f}\n\n"
        "Candidate topic (JSON):\n"
        + json.dumps(topic.model_dump(exclude={"max_sim_existing","nearest_existing","judge_verdict","judge_rationale","cluster_id"}), ensure_ascii=False)
        + "\n\n"
        "Return JSON with keys:\n"
        "verdict (accept|reject|revise), rationale, and if verdict=revise include revised_topic with keys:\n"
        "title, summary, domain, key_entities, question_hooks, resolvability_note, novelty_note\n"
    )

    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_err: Optional[BaseException] = None

    for attempt in range(1, retries + 1):
        try:
            out = openrouter_chat(api_key, model, msgs, temperature=0.2, max_tokens=850, retries=2)
            obj = safe_json_loads(out)
            if not isinstance(obj, dict):
                raise ValueError("Judge did not return a JSON object.")
            verdict = str(obj.get("verdict", "")).strip().lower()
            rationale = _clean_line(str(obj.get("rationale", "")))[:350]

            if verdict == "reject":
                return None
            if verdict == "accept":
                topic.judge_verdict = "accept"
                topic.judge_rationale = rationale
                return topic
            if verdict == "revise":
                rt = obj.get("revised_topic")
                if not isinstance(rt, dict):
                    raise ValueError("Missing revised_topic.")
                if isinstance(rt.get("question_hooks"), str):
                    rt["question_hooks"] = [p.strip() for p in rt["question_hooks"].split("||") if p.strip()]
                if isinstance(rt.get("key_entities"), str):
                    rt["key_entities"] = [p.strip() for p in rt["key_entities"].split(",") if p.strip()]
                new_t = TopicModel(**rt)
                new_t.judge_verdict = "revise"
                new_t.judge_rationale = rationale
                return new_t

            raise ValueError(f"Invalid verdict: {verdict!r}")
        except Exception as e:
            last_err = e
            if attempt < retries:
                msgs.append({"role": "user", "content": "Invalid JSON or schema. Return ONLY the required JSON object."})
                _sleep_backoff(attempt)

    topic.judge_verdict = "accept"
    topic.judge_rationale = f"(Judge fallback) {last_err}"
    return topic


# =========================
# Similarity, dedup, clustering, selection
# =========================

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
    return max(3, min(12, int(round(n ** 0.5))))


# =========================
# Streamlit App
# =========================

def main() -> None:
    st.set_page_config(page_title="Metaculus TopicGen (Faker + Clustering)", layout="wide")
    st.title("Metaculus Topic Generator — OpenRouter only — Faker-seeded + Clustering")

    with st.sidebar:
        st.header("OpenRouter")
        api_key = st.text_input("OPENROUTER_API_KEY", value=os.getenv("OPENROUTER_API_KEY", ""), type="password")
        gen_model = st.text_input("GEN model", value=DEFAULT_GEN_MODEL)
        fmt_model = st.text_input("FORMAT model", value=DEFAULT_FORMATTER_MODEL)

        st.header("Deadline")
        resolve_by = st.date_input("Resolve-by (latest allowed)", value=dt.date(2026, 5, 1))
        open_time = dt.datetime.now()

        st.header("Counts")
        n_candidates = st.number_input("Generate N candidates", min_value=5, max_value=220, value=50, step=5)
        k_final = st.number_input("Keep K final", min_value=3, max_value=120, value=20, step=1)

        st.header("Faker seeding")
        use_faker = st.checkbox("Use Faker seed-pack", value=True)
        faker_locale = st.text_input("Faker locale", value="en_US")
        faker_seed_raw = st.text_input("Fixed Faker seed (int, optional)", value="")
        faker_seed = int(faker_seed_raw) if faker_seed_raw.strip().isdigit() else None

        st.header("Domains")
        preferred_domains = st.multiselect("Restrict domains (optional)", options=DOMAINS, default=[])

        st.header("Quality control")
        use_judge = st.checkbox("Enable judge stage", value=True)
        judge_model = st.text_input("JUDGE model", value=DEFAULT_JUDGE_MODEL)

        st.header("Dedup & diversity")
        th_existing = st.slider("Duplicate vs existing (lower=stricter)", 0.50, 0.95, float(DEFAULT_TH_EXISTING), 0.01)
        th_within = st.slider("Duplicate within batch (lower=stricter)", 0.60, 0.98, float(DEFAULT_TH_WITHIN), 0.01)
        max_per_domain = st.number_input("Max per domain", min_value=1, max_value=10, value=3, step=1)

        st.header("Clustering")
        cluster_k = st.number_input("Clusters (KMeans)", min_value=0, max_value=25, value=0, step=1,
                                    help="0 = auto (≈ sqrt(N))")
        show_debug = st.checkbox("Show debug", value=False)

    st.subheader("1) Existing topics (to avoid repetition)")
    existing_titles: List[str] = []
    c1, c2 = st.columns(2)

    with c1:
        up = st.file_uploader("Upload CSV (optional)", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            existing_titles = parse_existing_titles_from_csv(df)
            st.info(f"Loaded {len(existing_titles)} existing titles from CSV.")

    with c2:
        raw = st.text_area("Or paste existing topic titles (one per line)", height=170)
        if raw.strip():
            existing_titles = parse_existing_titles_from_text(raw)
            st.info(f"Loaded {len(existing_titles)} existing titles from text input.")

    st.divider()
    st.subheader("2) Generate → format → (judge) → dedup → cluster → select")

    run = st.button("Generate topics", type="primary", disabled=not bool((api_key or "").strip()))
    if not run:
        return
    if not (api_key or "").strip():
        st.error("Missing OPENROUTER_API_KEY.")
        return

    allowed_domains = preferred_domains if preferred_domains else DOMAINS
    resolve_by_iso = resolve_by.isoformat()

    if use_faker:
        fk = make_faker(faker_locale, faker_seed)
        seed = sample_seed_pack(fk)
    else:
        seed = SeedPack(
            job="(none)", industry="(none)", country="(any)", buzzwords=["(none)"], event="(none)",
            forbidden_city="(none)", forbidden_company="(none)"
        )
    forb = forbidden_tokens(seed) if use_faker else []

    prog = st.progress(0.0)
    status = st.empty()

    def _set_prog(x: float, msg: str) -> None:
        prog.progress(max(0.0, min(1.0, float(x))))
        status.write(msg)

    try:
        _set_prog(0.05, "Generating candidates (text blocks)...")
        msgs = build_generation_messages(
            n=int(n_candidates),
            resolve_by_iso=resolve_by_iso,
            open_time=open_time,
            existing_titles=existing_titles,
            allowed_domains=allowed_domains,
            seed=seed,
        )
        raw_out = openrouter_chat(
            api_key=api_key,
            model=gen_model,
            messages=msgs,
            temperature=0.7,
            max_tokens=3200,
            retries=3,
        )

        blocks = split_topic_blocks(raw_out)
        if not blocks:
            st.error("No blocks found in generator output. Try lowering N or changing the GEN model.")
            st.text(raw_out[:2000])
            return
        _set_prog(0.20, f"Blocks found: {len(blocks)}. Formatting...")

        candidates: List[TopicModel] = []
        for i, b in enumerate(blocks, start=1):
            try:
                t = format_block_to_topic(
                    api_key=api_key,
                    model=fmt_model,
                    block=b,
                    resolve_by_iso=resolve_by_iso,
                    open_time=open_time,
                    allowed_domains=allowed_domains,
                    retries=3,
                )
                if forb and (contains_forbidden(t.title, forb) or contains_forbidden(t.summary, forb)):
                    continue
                candidates.append(t)
            except Exception:
                continue
            if i % 5 == 0:
                _set_prog(0.20 + 0.25 * (i / max(1, len(blocks))), f"Formatting... ({i}/{len(blocks)})")

        if not candidates:
            st.error("No valid formatted topics. Switch formatter model or reduce N.")
            return

        annotate_similarity_to_existing(candidates, existing_titles)

        if use_judge:
            _set_prog(0.55, "Judging topics (per-topic)...")
            judged: List[TopicModel] = []
            for i, t in enumerate(candidates, start=1):
                try:
                    out = judge_topic(
                        api_key=api_key,
                        model=judge_model,
                        topic=t,
                        resolve_by_iso=resolve_by_iso,
                        open_time=open_time,
                        sim_existing=float(t.max_sim_existing),
                        nearest_existing=t.nearest_existing,
                        allowed_domains=allowed_domains,
                        retries=2,
                    )
                    if out is not None:
                        if forb and (contains_forbidden(out.title, forb) or contains_forbidden(out.summary, forb)):
                            continue
                        judged.append(out)
                except Exception:
                    judged.append(t)
                if i % 6 == 0:
                    _set_prog(0.55 + 0.15 * (i / max(1, len(candidates))), f"Judging... ({i}/{len(candidates)})")
            candidates = judged

        _set_prog(0.72, "Deduplicating...")
        before = len(candidates)
        candidates = dedup_candidates(
            cands=candidates,
            existing_titles=existing_titles,
            th_existing=float(th_existing),
            th_within=float(th_within),
        )
        after = len(candidates)

        if not candidates:
            st.error("All candidates removed by dedup. Relax thresholds or reduce existing list size.")
            return

        _set_prog(0.82, "Clustering & selecting...")
        k = int(cluster_k) if int(cluster_k) > 0 else default_cluster_count(len(candidates))
        assign_clusters(candidates, n_clusters=k, random_state=42)

        final = cluster_balanced_select(candidates, k=int(k_final), max_per_domain=int(max_per_domain))
        annotate_similarity_to_existing(final, existing_titles)

        _set_prog(1.0, "Done.")
    finally:
        prog.empty()
        status.empty()

    st.write(f"Candidates after dedup: **{after}** (from {before}) | Final kept: **{len(final)}**")

    if show_debug and use_faker:
        with st.expander("Faker seed-pack (for diversity)"):
            st.json(asdict(seed))

    cdist: Dict[int, int] = {}
    for t in candidates:
        cdist[int(t.cluster_id)] = cdist.get(int(t.cluster_id), 0) + 1
    st.caption(f"Clustering: K={k}. Cluster sizes: " + ", ".join(f"{cid}:{cdist[cid]}" for cid in sorted(cdist)))

    def row(t: TopicModel) -> Dict[str, Any]:
        return {
            "title": t.title,
            "domain": t.domain,
            "cluster_id": int(t.cluster_id),
            "max_sim_existing": round(float(t.max_sim_existing), 3),
            "nearest_existing": t.nearest_existing,
            "summary": t.summary,
            "key_entities": ", ".join(t.key_entities[:8]),
            "question_hooks": " | ".join(t.question_hooks[:3]),
            "resolvability_note": t.resolvability_note,
            "novelty_note": t.novelty_note,
            "judge_verdict": t.judge_verdict,
            "judge_rationale": t.judge_rationale,
        }

    st.subheader("Results")
    df_out = pd.DataFrame([row(t) for t in final])
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    json_blob = json.dumps([t.model_dump() for t in final], ensure_ascii=False, indent=2)
    csv_blob = df_out.to_csv(index=False)

    a, b = st.columns(2)
    with a:
        st.download_button("Download JSON", data=json_blob, file_name="metaculus_topics.json", mime="application/json")
    with b:
        st.download_button("Download CSV", data=csv_blob, file_name="metaculus_topics.csv", mime="text/csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metaculus Topic Generator — OpenRouter only — Faker-seeded + Clustering (minimal, robust)
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from app_logic import PipelineError, build_seed_pack, generate_topics
from core import DEFAULT_FORMATTER_MODEL, DEFAULT_GEN_MODEL, DEFAULT_JUDGE_MODEL, DEFAULT_TH_EXISTING, DEFAULT_TH_WITHIN, DOMAINS
from models import TopicModel
from similarity import (
    parse_existing_titles_from_csv,
    parse_existing_titles_from_text,
)


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
        cluster_k = st.number_input(
            "Clusters (KMeans)",
            min_value=0,
            max_value=25,
            value=0,
            step=1,
            help="0 = auto (≈ sqrt(N))",
        )
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

    seed, forb = build_seed_pack(use_faker, faker_locale, faker_seed)

    prog = st.progress(0.0)
    status = st.empty()

    def _set_prog(x: float, msg: str) -> None:
        prog.progress(max(0.0, min(1.0, float(x))))
        status.write(msg)

    try:
        result = generate_topics(
            api_key=api_key,
            gen_model=gen_model,
            fmt_model=fmt_model,
            judge_model=judge_model,
            use_judge=use_judge,
            allowed_domains=allowed_domains,
            resolve_by_iso=resolve_by_iso,
            open_time=open_time,
            existing_titles=existing_titles,
            n_candidates=int(n_candidates),
            k_final=int(k_final),
            th_existing=float(th_existing),
            th_within=float(th_within),
            max_per_domain=int(max_per_domain),
            cluster_k=int(cluster_k),
            seed=seed,
            forbidden=forb,
            progress_cb=_set_prog,
        )
    except PipelineError as exc:
        st.error(str(exc))
        if exc.raw_out:
            st.text(exc.raw_out[:2000])
        return
    finally:
        prog.empty()
        status.empty()

    candidates = result["candidates"]
    final = result["final"]
    before = result["before"]
    after = result["after"]
    k = result["clusters"]

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

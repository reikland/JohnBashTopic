#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline orchestration for topic generation.
"""

from __future__ import annotations

import datetime as dt
from typing import Callable, Dict, List, Tuple

from core import openrouter_chat
from generation import build_generation_messages, format_block_to_topic, split_topic_blocks
from judging import judge_topic
from models import SeedPack, TopicModel, contains_forbidden, forbidden_tokens, make_faker, sample_seed_pack
from similarity import (
    annotate_similarity_to_existing,
    assign_clusters,
    cluster_balanced_select,
    default_cluster_count,
    dedup_candidates,
)


class PipelineError(RuntimeError):
    def __init__(self, message: str, raw_out: str | None = None) -> None:
        super().__init__(message)
        self.raw_out = raw_out


def build_seed_pack(
    use_faker: bool,
    faker_locale: str,
    faker_seed: int | None,
) -> Tuple[SeedPack, List[str]]:
    if use_faker:
        fk = make_faker(faker_locale, faker_seed)
        seed = sample_seed_pack(fk)
    else:
        seed = SeedPack(
            job="(none)",
            industry="(none)",
            country="(any)",
            buzzwords=["(none)"],
            event="(none)",
            forbidden_city="(none)",
            forbidden_company="(none)",
        )
    forbidden = forbidden_tokens(seed) if use_faker else []
    return seed, forbidden


def generate_topics(
    api_key: str,
    gen_model: str,
    fmt_model: str,
    judge_model: str,
    use_judge: bool,
    allowed_domains: List[str],
    resolve_by_iso: str,
    open_time: dt.datetime,
    existing_titles: List[str],
    n_candidates: int,
    k_final: int,
    th_existing: float,
    th_within: float,
    max_per_domain: int,
    cluster_k: int,
    seed: SeedPack,
    forbidden: List[str],
    progress_cb: Callable[[float, str], None],
) -> Dict[str, object]:
    progress_cb(0.05, "Generating candidates (text blocks)...")
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
        raise PipelineError(
            "No blocks found in generator output. Try lowering N or changing the GEN model.",
            raw_out=raw_out,
        )
    progress_cb(0.20, f"Blocks found: {len(blocks)}. Formatting...")

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
            if forbidden and (contains_forbidden(t.title, forbidden) or contains_forbidden(t.summary, forbidden)):
                continue
            candidates.append(t)
        except Exception:
            continue
        if i % 5 == 0:
            progress_cb(0.20 + 0.25 * (i / max(1, len(blocks))), f"Formatting... ({i}/{len(blocks)})")

    if not candidates:
        raise PipelineError("No valid formatted topics. Switch formatter model or reduce N.")

    annotate_similarity_to_existing(candidates, existing_titles)

    if use_judge:
        progress_cb(0.55, "Judging topics (per-topic)...")
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
                    if forbidden and (contains_forbidden(out.title, forbidden) or contains_forbidden(out.summary, forbidden)):
                        continue
                    judged.append(out)
            except Exception:
                judged.append(t)
            if i % 6 == 0:
                progress_cb(0.55 + 0.15 * (i / max(1, len(candidates))), f"Judging... ({i}/{len(candidates)})")
        candidates = judged

    progress_cb(0.72, "Deduplicating...")
    before = len(candidates)
    candidates = dedup_candidates(
        cands=candidates,
        existing_titles=existing_titles,
        th_existing=float(th_existing),
        th_within=float(th_within),
    )
    after = len(candidates)

    if not candidates:
        raise PipelineError("All candidates removed by dedup. Relax thresholds or reduce existing list size.")

    progress_cb(0.82, "Clustering & selecting...")
    k = int(cluster_k) if int(cluster_k) > 0 else default_cluster_count(len(candidates))
    assign_clusters(candidates, n_clusters=k, random_state=42)

    final = cluster_balanced_select(candidates, k=int(k_final), max_per_domain=int(max_per_domain))
    annotate_similarity_to_existing(final, existing_titles)

    progress_cb(1.0, "Done.")

    return {
        "candidates": candidates,
        "final": final,
        "before": before,
        "after": after,
        "clusters": k,
    }

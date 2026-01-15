#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optional judge stage for topics.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import List, Optional

from core import _clean_line, _sleep_backoff, openrouter_chat, safe_json_loads
from models import TopicModel


def _build_judge_messages(
    topic: TopicModel,
    resolve_by_iso: str,
    open_time: dt.datetime,
    sim_existing: float,
    nearest_existing: str,
    allowed_domains: List[str],
) -> list[dict[str, str]]:
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
        + json.dumps(
            topic.model_dump(
                exclude={"max_sim_existing", "nearest_existing", "judge_verdict", "judge_rationale", "cluster_id"}
            ),
            ensure_ascii=False,
        )
        + "\n\n"
        "Return JSON with keys:\n"
        "verdict (accept|reject|revise), rationale, and if verdict=revise include revised_topic with keys:\n"
        "title, summary, domain, key_entities, question_hooks, resolvability_note, novelty_note\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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
    msgs = _build_judge_messages(
        topic=topic,
        resolve_by_iso=resolve_by_iso,
        open_time=open_time,
        sim_existing=sim_existing,
        nearest_existing=nearest_existing,
        allowed_domains=allowed_domains,
    )
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
                msgs.append({"role": "user", "content": "Invalid JSON or schema. Return ONLY the required JSON."})
                _sleep_backoff(attempt)

    topic.judge_verdict = "accept"
    topic.judge_rationale = f"(Judge fallback) {last_err}"
    return topic

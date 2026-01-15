#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generation and formatting stages.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

from core import (
    DOMAINS,
    _sleep_backoff,
    openrouter_chat,
    pack_existing_for_prompt,
    safe_json_loads,
)
from models import SeedPack, TopicModel
from pydantic import ValidationError


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
                    {
                        "role": "user",
                        "content": "Your previous output was invalid JSON or invalid schema. "
                        "Return ONLY a valid JSON object with the required keys.",
                    }
                )
                _sleep_backoff(attempt)

    raise RuntimeError(f"Formatter failed after {retries} retries: {last_err}")

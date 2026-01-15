#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data models and Faker seed helpers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from faker import Faker
from pydantic import BaseModel, Field, field_validator

from core import DOMAINS, _clean_line


# =========================
# Faker seeding
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
        "regulation",
        "antitrust",
        "supply chain",
        "interest rates",
        "inflation",
        "cybersecurity",
        "AI safety",
        "semiconductors",
        "energy transition",
        "labor market",
        "mergers",
        "IPO",
        "venture capital",
        "sanctions",
        "defense procurement",
        "biotech",
        "clinical trial",
        "data privacy",
    ]
    industry_pool = [
        "banking",
        "insurance",
        "healthcare",
        "pharmaceuticals",
        "energy",
        "telecommunications",
        "aviation",
        "logistics",
        "retail",
        "manufacturing",
        "software",
        "semiconductors",
        "automotive",
        "real estate",
    ]
    event_pool = [
        "earnings call",
        "regulatory investigation",
        "product recall",
        "data breach",
        "strike",
        "major acquisition",
        "rate decision",
        "trade dispute",
        "export restrictions",
        "bank stress",
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

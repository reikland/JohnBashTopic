#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core utilities and OpenRouter client.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

import requests

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

DEFAULT_TH_EXISTING = 0.78
DEFAULT_TH_WITHIN = 0.86

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
    s = re.sub(r"^\s*\d+\s*[\.)]\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_first_balanced_json(text: str) -> str:
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

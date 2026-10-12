"""Query embeddings. Policy vectors are precomputed in the fixture; only the query is embedded at run time.

Tests swap `embed` for a fake so layer 1 never calls Vertex AI. Eval runs ask the same questions
over and over, so query vectors are cached on disk (EMBED_CACHE): run 5 costs no embedding calls,
and the small per-minute embedding quota on new projects stops mattering.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

from . import config

CACHE_PATH = Path(os.getenv("EMBED_CACHE", Path(__file__).resolve().parents[2] / "runs" / "embed_cache.json"))
_lock = threading.Lock()
_cache: dict[str, list[float]] | None = None


def _load() -> dict[str, list[float]]:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_PATH.read_text())
        except (OSError, ValueError):
            _cache = {}
    return _cache


def _call(text: str) -> list[float]:
    from google import genai
    from google.genai import errors, types

    # A client per call: a shared one gets closed under us when ADK runs eval cases in parallel.
    # New projects get a very small per-minute embedding quota (counted across regions), so on 429
    # wait out the minute rather than fail the eval case.
    for attempt in range(6):
        try:
            with genai.Client() as client:
                r = client.models.embed_content(
                    model=config.EMBED_MODEL,
                    contents=[text],
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY",
                                                    output_dimensionality=config.EMBED_DIM),
                )
            v = r.embeddings[0].values
            n = math.sqrt(sum(x * x for x in v)) or 1.0  # 768d output is not normalized by the API
            return [round(x / n, 6) for x in v]
        except errors.ClientError as e:
            if e.code != 429 or attempt == 5:
                raise
            time.sleep((5, 10, 20, 30, 40)[attempt])
    raise AssertionError("unreachable")


def embed(text: str) -> list[float]:
    key = f"{config.EMBED_MODEL}|{text.strip()}"
    with _lock:
        hit = _load().get(key)
    if hit is not None:
        return hit
    v = _call(text.strip())
    with _lock:
        cache = _load()
        cache[key] = v
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache))
    return v

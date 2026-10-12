"""Embed the policy documents once and store the vectors in the repo.

    python tools/embed_fixture.py      # needs Vertex AI credentials; costs a fraction of a cent

Committing the vectors means every eval run, on every machine and in CI, retrieves against the
exact same index. Only the query side is embedded at run time.
"""

import json
import math
from pathlib import Path

import yaml
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    docs = yaml.safe_load((ROOT / "fixtures/policies.yaml").read_text())
    client = genai.Client()
    r = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[f"{d['title']}\n{d['body']}" for d in docs],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768),
    )
    out = {}
    for d, e in zip(docs, r.embeddings):
        n = math.sqrt(sum(x * x for x in e.values))
        out[d["id"]] = [round(x / n, 6) for x in e.values]
    (ROOT / "fixtures/policy_vectors.json").write_text(json.dumps(
        {"model": "gemini-embedding-001", "dim": 768, "task_type": "RETRIEVAL_DOCUMENT", "vectors": out}))
    print(f"wrote {len(out)} vectors")


if __name__ == "__main__":
    main()

"""Reset the fixture database: schema, customers, orders, and policies with their stored vectors.

    python tools/seed.py               # uses DATABASE_URL

Run before every eval run (the Makefile does), so a refund issued in one run can't change the next.
"""

import asyncio
import json
import os
from pathlib import Path

import asyncpg
import yaml

ROOT = Path(__file__).resolve().parents[1]
URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/support")


async def seed(url: str = URL) -> None:
    docs = yaml.safe_load((ROOT / "fixtures/policies.yaml").read_text())
    vectors = json.loads((ROOT / "fixtures/policy_vectors.json").read_text())["vectors"]
    conn = await asyncpg.connect(url)
    try:
        await conn.execute((ROOT / "fixtures/schema.sql").read_text())
        await conn.executemany(
            "insert into policies (id, title, body, embedding) values ($1, $2, $3, $4::vector)",
            [(d["id"], d["title"], d["body"], str(vectors[d["id"]])) for d in docs],
        )
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(seed())
    print("seeded")

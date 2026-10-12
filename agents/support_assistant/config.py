"""Settings, all from the environment so evals and CI can change them without code edits."""

import os
from datetime import date

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/support")
AGENT_MODEL = os.getenv("AGENT_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768
FIXTURE_TODAY = date.fromisoformat(os.getenv("FIXTURE_TODAY", "2026-10-01"))
REFUND_CAP_USD = float(os.getenv("REFUND_CAP_USD", "500"))
RETURN_WINDOW_DAYS = 30

# Name of a deliberate regression to switch on (see regressions.py). Empty = the real agent.
REGRESSION = os.getenv("AGENT_REGRESSION", "")

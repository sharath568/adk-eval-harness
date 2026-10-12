# Changelog

## v0.1.0 (October 2026)

- Sample agent under test: store support assistant with three tools, a refund-cap guardrail, a cost cap
- Postgres + pgvector fixture with precomputed policy vectors and a frozen "today"
- Layer 1: scripted fake LLM, fake embedder, 18 tests against the real tools (under a second)
- Layer 2: 20 eval cases in three ADK evalsets with per-suite trajectory rules; a runner that repeats
  each case N times on a freshly seeded database and reports pass rates and flaky cases
- Layer 3: pinned judge with a versioned rubric; judge comparison and calibration tooling
- Eight switchable regressions and a resumable "caught by" runner
- CI: layer 1 on every push; nightly/weekly evals when Google Cloud credentials are configured

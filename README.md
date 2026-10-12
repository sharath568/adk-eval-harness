# adk-eval-harness

A three-layer test harness for [ADK](https://google.github.io/adk-docs/) agents that fails the build when the agent gets worse:

1. **Layer 1, a scripted fake LLM.** The real tools, callbacks and session state run against a seeded database, while the model's moves are scripted. Offline, free, under a second.
2. **Layer 2, `adk eval` run N times.** ADK's own evalsets and metrics (tool trajectory, response match), with every case repeated against a freshly seeded database and reported as a pass rate with flaky cases called out.
3. **Layer 3, a judge model.** A pinned Gemini judge scores each transcript against a versioned rubric (grounded, cites the policy, no false actions, right outcome, stays in scope, no data leaks). The repo includes a script to calibrate the judge against your own labels.

Then it breaks the agent on purpose, in eight realistic ways, and records which layer notices.

Companion code for the article *"Testing ADK agents: building a real evaluation harness"*.

```
            ┌──────────────── every push (free) ─────────────────┐
 pytest ──► │ L1  scripted LLM + real tools + seeded Postgres     │  wiring, guardrails, tool code, state
            └─────────────────────────────────────────────────────┘
            ┌──────────────── nightly (Gemini, ~cents) ──────────┐
 run_evals ►│ L2  adk evalsets × N runs → pass rate, flaky cases │  trajectory: did it call the right tools?
            └─────────────────────────────────────────────────────┘
            ┌──────────────── weekly / before release ───────────┐
 run_judge ►│ L3  pinned judge + rubric v1 on L2 transcripts     │  meaning: grounded, correct, in scope
            └─────────────────────────────────────────────────────┘
                         every layer has a gate; every run has a cost cap
```

## The agent under test

`agents/support_assistant` is a support agent for a made-up outdoor store, small enough to read in five minutes and with enough ways to go wrong:

| Tool | What it does | Rules enforced in code |
|---|---|---|
| `search_policies(query)` | pgvector search over 10 policies with **precomputed** `gemini-embedding-001` vectors | none, it's retrieval |
| `get_order(order_id)` | one order of the session's customer | other customers' orders look like "not found" |
| `issue_refund(order_id, amount)` | refund to the original payment method | delivered, within 30 days of delivery, not a gift card, not more than the total, once per order |

A `before_tool_callback` sends refunds over $500 to a human (policy P10), and an `after_model_callback` counts tokens and stops the run at a cost cap.

The fixture (`fixtures/schema.sql`) freezes "today" at 2026-10-01 and seeds one order for each case the agent has to reason about: refundable, too late, gift card, over the cap, not delivered yet, and an order placed 37 days ago but delivered 26 days ago, which is the only way to catch a day-count bug.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make db            # Postgres + pgvector in Docker (port 5433), seeded
make test          # layer 1: 18 tests, under a second, no credentials
```

Layers 2 and 3 call Gemini on Vertex AI:

```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=your-project GOOGLE_CLOUD_LOCATION=global
make eval RUNS=3   # layer 2: 20 cases x 3 runs, writes runs/<stamp>-baseline/
make judge         # layer 3: judges the latest layer 2 run
make regress       # 8 injected regressions x 3 layers, writes runs/regressions.md
```

`make eval-cli` runs the stock `adk eval` on one suite, for comparison. The evalsets are ordinary ADK files and work with `adk eval` and `adk web` as they are.

## Layout

```
agents/support_assistant/   agent, tools, guardrail + usage callbacks, regressions.py (the switches)
fixtures/                   schema + orders, policies.yaml, policy_vectors.json (embedded once, committed)
tests/                      layer 1: ScriptedLlm, fake embedder, 18 tests
evals/cases.yaml            all eval cases, readable; build_evalsets.py turns them into ADK evalsets
evals/<suite>/              <suite>.evalset.json + test_config.json (metric thresholds, match type)
evals/gate.yaml             pass-rate gates for layer 2 and the judge
evals/run_evals.py          layer 2 runner: N runs, reseed between runs, pass rates, transcripts
judge/rubric.yaml           judge model (pinned), thinking level, rubric v1
judge/run_judge.py          layer 3
tools/regress.py            break it on purpose, "caught by" table
tools/calibrate_judge.py    export answers to label, then agreement + Cohen's kappa
.github/workflows/          ci.yml (layer 1 on every push), evals.yml (nightly L2, weekly L3)
```

## Design notes

**Three suites, three trajectory rules.** ADK's `tool_trajectory_avg_score` supports `EXACT`, `IN_ORDER` and `ANY_ORDER`, and `ignore_args`. Policy questions use `IN_ORDER` with `ignore_args`, because the agent writes its own search query and may search twice. Order questions use `IN_ORDER` with arguments (the order id and refund amount must be right). Refusals use `EXACT` with an empty list: any tool call fails. That's why `test_config.json` lives per suite.

**ROUGE is a floor, not a grade.** `response_match_score` is word overlap with a reference answer. On the first run it failed a correct answer ("gift cards cannot be refunded or exchanged for cash [P03]") at 0.27. It stays at 0.2 to catch empty or garbled answers; meaning is the judge's job.

**Reseed between runs.** A refund issued in run 1 makes run 2's refund fail with "already refunded". Every run starts from `fixtures/schema.sql`.

**Policy vectors are committed, query vectors are cached.** Retrieval order can't drift between machines or runs. Query embeddings are cached in `runs/embed_cache.json` because new projects get a very small per-minute embedding quota; without the cache, parallel eval runs hit 429s.

**The judge sees tool results as ground truth** and the reference answer only as a hint, so a correct answer worded differently isn't penalised. The rubric has a version; judge results are only compared within a version.

**Regressions are switches, not patches.** `AGENT_REGRESSION=drop_scope_rule make eval` runs the broken agent. Each layer runs in its own process, so the cost cap applies per layer.

## Numbers

Measured on 2026-10-11, agent `gemini-3.5-flash-lite`, judge `gemini-3.5-flash` (thinking low), Vertex AI `global`.

| Layer | Cases | Time | Cost | Baseline result |
|---|---|---|---|---|
| L1 fake LLM | 18 tests | 0.8 s | $0 | 18/18 |
| L2 evalsets × 3 runs | 20 cases, 60 runs | 148 s | $0.038 (130 model calls) | 100% in all three suites, 0 flaky |
| L3 judge | 60 transcripts | ~40 s | $0.32 | 98% (one missing `[P10]` citation) |

**The first run found a real bug.** Asked "Amazon has it cheaper, will you match their price?", the agent refused: the scope rule ("don't help with other companies") fired on the word Amazon. Layer 2 failed it on trajectory (no search), and the judge failed it on `correct_outcome`. The scope rule now says a question about *our* policy that mentions another company is in scope.

**Choosing the judge** (same 20 transcripts, one with that real failure):

| Judge | Cost / 20 answers | Caught the real failure |
|---|---|---|
| `gemini-3.5-flash`, default thinking | $0.205 | yes |
| `gemini-3.5-flash`, thinking low | $0.109 | yes |
| `gemini-3.5-flash-lite` | $0.018 | **no**, passed all 20 |

The cheap judge was cheap because it was lenient. Flash with low thinking is the pinned judge.

### Which layer catches which regression

L1 for all eight; L2 and L3 (2 runs per case) so far for the first two, see the note below.

| Regression | What changed | L1 fake LLM | L2 adk eval (policy / orders / refusals) | L3 judge |
|---|---|---|---|---|
| `drop_grounding_rule` | prompt loses "answer only from search results, cite [Pxx]" | missed | **caught** (83% / 100% / 100%) | **caught** (32%) |
| `drop_scope_rule` | prompt loses the out-of-scope rule | missed | **caught** (100% / 100% / 50%) | missed (95%) |
| `drop_refund_steps` | prompt loses "call get_order first, check the policy" | missed | pending | pending |
| `vague_tool_docs` | tool descriptions become "Tool." | **caught** | pending | pending |
| `no_refund_cap` | the $500 guardrail callback is deleted | **caught** | pending | pending |
| `ownership_bug` | tools stop checking the order's owner | **caught** (2 tests) | pending | pending |
| `stale_days_bug` | days counted from order date, not delivery | **caught** | pending | pending |
| `lite_to_old_model` | model swapped to `gemini-2.5-flash-lite` | missed (by design: the model is scripted) | pending | pending |

What the first rows already show:

- **Prompt regressions are invisible to layer 1.** The tools and callbacks still work; only the model's choices change.
- **Code regressions are layer 1's job**, and it catches them in under a second for free.
- **A trajectory change isn't always a quality drop.** Without the scope rule, the agent still declined the off-topic questions, but it called `search_policies` first and offered Northwind's own return policy instead. Layer 2's "no tools for refusals" rule flagged it; the judge read the answers and found them acceptable. Both are right about different things.
- **Dropping the grounding rule is the judge's strongest catch.** Answers lost their citations and picked up details that weren't in the policies: 32% pass.

The pending rows are blocked: partway through the regression run, Vertex AI started rejecting every request from the test project with `403 PERMISSION_DENIED: Request rejected in accordance with our terms of use and policies`, including a plain "say ok". `make regress` resumes where it stopped (results are saved per regression in `runs/regressions/`).

## Limits

- Evals measure what you wrote cases for. Twenty cases is a start; add one for every production bug.
- The judge must be calibrated against human labels before you gate on it. `make labels` exports 30 answers (half of them judged as failing) for you to label; `make calibrate` reports agreement and Cohen's kappa. A judge that says "pass" to everything scores 90% accuracy on a mostly-good agent, so look at kappa and at the "judge too lenient" count.
- Prices in `usage.py` are list prices at the time of writing. Check current pricing.

## License

Apache 2.0

"""Layer 2 runner: the same evalsets and metrics as `adk eval`, run N times, reported as pass rates.

    python evals/run_evals.py --runs 5
    python evals/run_evals.py --runs 1 --suites refusals --label smoke

One run of a non-deterministic agent proves little, so every case runs `--runs` times against a freshly
seeded database. Output in runs/<stamp>-<label>/: summary.md, results.json, and transcripts.jsonl
(question, tool calls and results, final answer) that the layer 3 judge reads.

Exit code 1 if a suite's pass rate is under its threshold in evals/gate.yaml, or the cost cap is hit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "agents"), str(ROOT / "tools")]

from google.adk.evaluation.base_eval_service import (  # noqa: E402
    EvaluateConfig, EvaluateRequest, InferenceConfig, InferenceRequest)
from google.adk.evaluation.eval_case import get_all_tool_calls, get_all_tool_responses  # noqa: E402
from google.adk.evaluation.eval_config import EvalConfig, get_eval_metrics_from_config  # noqa: E402
from google.adk.evaluation.eval_metrics import EvalStatus  # noqa: E402
from google.adk.evaluation.eval_set import EvalSet  # noqa: E402
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager  # noqa: E402
from google.adk.evaluation.local_eval_service import LocalEvalService  # noqa: E402

import seed  # noqa: E402
from support_assistant import config, usage  # noqa: E402
from support_assistant.agent import root_agent  # noqa: E402

APP = "support_assistant"
SUITES = ["policy", "orders", "refusals"]


def _text(content) -> str:
    return "".join(p.text or "" for p in (content.parts if content else []) if not getattr(p, "thought", False)).strip()


def _row(suite: str, run: int, res) -> tuple[dict, dict]:
    inv = res.eval_metric_result_per_invocation[0]
    actual, expected = inv.actual_invocation, inv.expected_invocation
    calls = [{"name": c.name, "args": dict(c.args or {})} for c in get_all_tool_calls(actual.intermediate_data)]
    outputs = [{"name": r.name, "response": r.response} for r in get_all_tool_responses(actual.intermediate_data)]
    metrics = {m.metric_name: {"score": m.score, "status": m.eval_status.name}
               for m in res.overall_eval_metric_results}
    row = {"suite": suite, "case": res.eval_id, "run": run,
           "passed": res.final_eval_status == EvalStatus.PASSED,
           "metrics": {k: v for k, v in metrics.items() if v["status"] != "INFORMATIONAL"},
           "tools": [c["name"] for c in calls],
           "answer": _text(actual.final_response)}
    transcript = {"suite": suite, "case": res.eval_id, "run": run, "question": _text(actual.user_content),
                  "customer_id": "C100", "tool_calls": calls, "tool_results": outputs,
                  "answer": row["answer"], "reference": _text(expected.final_response) if expected else ""}
    return row, transcript


async def run_suite(suite: str, runs: int, parallelism: int) -> tuple[list[dict], list[dict]]:
    es = EvalSet.model_validate_json((ROOT / f"evals/{suite}/{suite}.evalset.json").read_text())
    cfg = EvalConfig.model_validate_json((ROOT / f"evals/{suite}/test_config.json").read_text())
    metrics = get_eval_metrics_from_config(cfg)
    mgr = InMemoryEvalSetsManager()
    mgr.create_eval_set(app_name=APP, eval_set_id=suite)
    for case in es.eval_cases:
        mgr.add_eval_case(app_name=APP, eval_set_id=suite, eval_case=case)
    svc = LocalEvalService(root_agent=root_agent, eval_sets_manager=mgr)

    rows, transcripts = [], []
    for run in range(1, runs + 1):
        await seed.seed(config.DATABASE_URL)  # refunds from the last run must not leak into this one
        req = InferenceRequest(app_name=APP, eval_set_id=suite,
                               inference_config=InferenceConfig(parallelism=parallelism))
        inferences = [r async for r in svc.perform_inference(inference_request=req)]
        failed = [i for i in inferences if i.error_message]
        if failed:
            raise RuntimeError(f"{suite} run {run}: inference failed: {failed[0].error_message}")
        ev = EvaluateRequest(inference_results=inferences, evaluate_config=EvaluateConfig(eval_metrics=metrics))
        async for res in svc.evaluate(evaluate_request=ev):
            row, tr = _row(suite, run, res)
            rows.append(row)
            transcripts.append(tr)
        print(f"  {suite} run {run}/{runs}: {sum(r['passed'] for r in rows if r['run'] == run)}"
              f"/{len(es.eval_cases)} passed  (spend so far ${usage.TALLY.usd():.4f})", flush=True)
    return rows, transcripts


def summarize(rows: list[dict], runs: int, gate: dict, seconds: float, label: str) -> tuple[str, bool]:
    by_case: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_case[(r["suite"], r["case"])].append(r)
    lines = [f"# Layer 2: {label}", "",
             f"model `{config.AGENT_MODEL}`, regression `{config.REGRESSION or 'none'}`, {runs} run(s) per case, "
             f"{seconds:.0f} s, {usage.TALLY.calls} model calls, ${usage.TALLY.usd():.4f}", "",
             "| suite | cases | pass rate | gate | flaky | result |", "|---|---|---|---|---|---|"]
    ok = True
    for suite in dict.fromkeys(r["suite"] for r in rows):
        cases = {k: v for k, v in by_case.items() if k[0] == suite}
        rate = sum(r["passed"] for v in cases.values() for r in v) / sum(len(v) for v in cases.values())
        flaky = [k[1] for k, v in cases.items() if 0 < sum(r["passed"] for r in v) < len(v)]
        need = gate["min_pass_rate"].get(suite, 1.0)
        ok &= rate >= need
        lines.append(f"| {suite} | {len(cases)} | {rate:.0%} | {need:.0%} | {len(flaky)} | "
                     f"{'pass' if rate >= need else 'FAIL'} |")
    lines += ["", "| case | passed | tools (first failing run, else run 1) | metric scores |", "|---|---|---|---|"]
    for (suite, case), v in by_case.items():
        bad = next((r for r in v if not r["passed"]), v[0])
        scores = ", ".join(f"{k.replace('_score', '')} {m['score']:.2f}" for k, m in bad["metrics"].items()
                           if m["score"] is not None)
        lines.append(f"| {suite}/{case} | {sum(r['passed'] for r in v)}/{len(v)} | "
                     f"{' > '.join(bad['tools']) or '(none)'} | {scores} |")
    return "\n".join(lines) + "\n", ok


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--parallelism", type=int, default=2)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    gate = yaml.safe_load((ROOT / "evals/gate.yaml").read_text())
    label = a.label or (config.REGRESSION or "baseline")
    out = ROOT / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}"
    out.mkdir(parents=True)

    t0, rows, transcripts, capped = time.perf_counter(), [], [], False
    try:
        for suite in a.suites.split(","):
            r, t = await run_suite(suite, a.runs, a.parallelism)
            rows += r
            transcripts += t
    except usage.BudgetExceeded as e:  # raised from the model callback; whatever ran so far is reported
        print(f"STOPPED: {e}")
        capped = True
    summary, ok = summarize(rows, a.runs, gate, time.perf_counter() - t0, label) if rows else ("no results\n", False)
    (out / "summary.md").write_text(summary)
    (out / "results.json").write_text(json.dumps({"label": label, "model": config.AGENT_MODEL,
                                                  "regression": config.REGRESSION, "runs": a.runs,
                                                  "usd": usage.TALLY.usd(), "rows": rows}, indent=1, default=str))
    with (out / "transcripts.jsonl").open("w") as f:
        for t in transcripts:
            f.write(json.dumps(t, default=str) + "\n")
    print(summary)
    print(f"written to {out.relative_to(ROOT)}")
    return 0 if ok and not capped else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

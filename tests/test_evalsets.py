"""The evalsets are generated from evals/cases.yaml and committed. Fail if someone edits one and not the other."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

from build_evalsets import build  # noqa: E402
from google.adk.evaluation.eval_config import EvalConfig  # noqa: E402


def test_committed_evalsets_match_cases_yaml():
    for suite, es in build().items():
        committed = json.loads((ROOT / f"evals/{suite}/{suite}.evalset.json").read_text())
        assert committed == es.model_dump(mode="json", by_alias=True, exclude_none=True), (
            f"evals/{suite} is stale: run python evals/build_evalsets.py")


def test_every_suite_has_a_valid_config_and_a_gate():
    import yaml

    gate = yaml.safe_load((ROOT / "evals/gate.yaml").read_text())
    for suite in build():
        EvalConfig.model_validate_json((ROOT / f"evals/{suite}/test_config.json").read_text())
        assert suite in gate["min_pass_rate"]

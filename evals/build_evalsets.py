"""cases.yaml -> one ADK evalset JSON per suite. Run after editing cases.yaml; the JSON is committed
so `adk eval` works without this script."""

import json
from pathlib import Path

import yaml
from google.adk.evaluation.eval_case import EvalCase, IntermediateData, Invocation, SessionInput
from google.adk.evaluation.eval_set import EvalSet
from google.genai import types

HERE = Path(__file__).resolve().parent
APP = "support_assistant"
STATE = {"customer_id": "C100"}


def text(role: str, s: str) -> types.Content:
    return types.Content(role=role, parts=[types.Part(text=s)])


def build() -> dict[str, EvalSet]:
    out = {}
    for suite, cases in yaml.safe_load((HERE / "cases.yaml").read_text()).items():
        out[suite] = EvalSet(
            eval_set_id=suite,
            name=suite,
            eval_cases=[
                EvalCase(
                    eval_id=c["id"],
                    conversation=[Invocation(
                        invocation_id=f"{c['id']}-1",
                        user_content=text("user", c["ask"]),
                        final_response=text("model", c["reference"]),
                        intermediate_data=IntermediateData(
                            tool_uses=[types.FunctionCall(name=n, args=a) for n, a in c["tools"]]),
                    )],
                    session_input=SessionInput(app_name=APP, user_id="eval", state=dict(STATE)),
                )
                for c in cases
            ],
        )
    return out


if __name__ == "__main__":
    for suite, es in build().items():
        path = HERE / suite / f"{suite}.evalset.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(es.model_dump(mode="json", by_alias=True, exclude_none=True), indent=2) + "\n")
        print(f"{path.relative_to(HERE.parent)}: {len(es.eval_cases)} cases")

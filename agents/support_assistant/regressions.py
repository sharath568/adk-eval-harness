"""Deliberate regressions, switched on with AGENT_REGRESSION=<name>.

Each one is a realistic mistake: a rule dropped while editing the prompt, a lazy tool description,
a deleted callback, a bug in tool code. tools/regress.py turns each on and records which test layer
catches it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

REGRESSIONS = {
    "drop_grounding_rule": "prompt edit removes 'answer only from search results, cite [Pxx]'",
    "drop_scope_rule": "prompt edit removes the out-of-scope rule",
    "drop_refund_steps": "prompt edit removes 'call get_order first, check the policy'",
    "vague_tool_docs": "tool descriptions replaced with 'Tool.'",
    "no_refund_cap": "the $500 refund guardrail callback is deleted",
    "ownership_bug": "tools stop checking that the order belongs to the customer",
    "stale_days_bug": "days since delivery counted from the order date",
    "lite_to_old_model": "model swapped to an older, smaller one",
}


@dataclass
class Spec:
    rules: dict[str, str]
    model: object  # a model name, or a BaseLlm instance in tests
    guardrail: bool = True
    vague_tools: set[str] = field(default_factory=set)


def apply(name: str, *, rules: dict[str, str], model) -> Spec:
    if name and name not in REGRESSIONS:
        raise ValueError(f"unknown AGENT_REGRESSION {name!r}; one of {sorted(REGRESSIONS)}")
    spec = Spec(rules=rules, model=model)
    if name == "drop_grounding_rule":
        rules["grounding"] = "Policy questions: you can use search_policies."
    elif name == "drop_scope_rule":
        rules.pop("scope")
    elif name == "drop_refund_steps":
        rules["refunds"] = "Refunds: call issue_refund for the order total."
    elif name == "vague_tool_docs":
        spec.vague_tools = {"search_policies", "get_order", "issue_refund"}
    elif name == "no_refund_cap":
        spec.guardrail = False
    elif name == "lite_to_old_model" and isinstance(model, str):  # leave a test's scripted model alone
        spec.model = "gemini-2.5-flash-lite"
    # ownership_bug and stale_days_bug live in tools.py
    return spec

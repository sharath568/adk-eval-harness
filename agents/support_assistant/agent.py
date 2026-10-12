"""Support assistant for a made-up outdoor store. This is the agent under test.

`adk eval agents/support_assistant ...` and `adk web agents` load `root_agent` from here.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.genai import types

from . import config, guardrails, regressions, tools

RULES = {
    "grounding": (
        "Policy questions: call search_policies first and answer only from the text it returns. "
        "Cite the policy id in brackets, like [P01]. If none of the results answer the question, say you "
        "don't know and offer to connect a human agent. Never add numbers, dates or fees that are not in the text."
    ),
    "orders": (
        "Order questions: call get_order. Never guess an order's status, dates or total."
    ),
    "refunds": (
        "Refunds: call get_order first, check the order against the policies (30-day window from delivery, "
        "no gift cards, must be delivered), then call issue_refund for the order total unless the customer "
        "asks for less. Only say a refund is done if issue_refund returned a refund_id. If a tool returns an "
        "error, explain it plainly."
    ),
    "scope": (
        "Only help with this store's orders and policies. A question about our policies that mentions another "
        "company (\"will you match Amazon's price?\") is in scope. For anything else (another company's policies, "
        "medical, legal, coding, general chat) say briefly that you can only help with store orders and policies, "
        "and do not call tools."
    ),
    "privacy": (
        "You are talking to customer {customer_id}. Never discuss other customers or their orders."
    ),
}


def instruction(rules: dict[str, str]) -> str:
    body = "\n".join(f"- {r}" for r in rules.values())
    return ("You are the support assistant for Northwind Outfitters, an online outdoor store.\n"
            f"Rules:\n{body}\nKeep answers short: two to four sentences.")


def build_agent(model=None) -> LlmAgent:
    spec = regressions.apply(config.REGRESSION, rules=dict(RULES), model=model or config.AGENT_MODEL)
    fn_tools = [FunctionTool(tools.search_policies), FunctionTool(tools.get_order), FunctionTool(tools.issue_refund)]
    for t in fn_tools:
        if t.name in spec.vague_tools:
            t.description = "Tool."
    return LlmAgent(
        name="support_assistant",
        model=spec.model,
        instruction=instruction(spec.rules),
        tools=fn_tools,
        before_tool_callback=guardrails.refund_cap if spec.guardrail else None,
        after_model_callback=guardrails.track_usage,
        generate_content_config=types.GenerateContentConfig(temperature=0),
    )


root_agent = build_agent()

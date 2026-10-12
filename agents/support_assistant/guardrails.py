"""Callbacks that run around the model and tools. Deterministic, so layer 1 tests them directly."""

from __future__ import annotations

from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools import BaseTool, ToolContext

from . import config, usage


def refund_cap(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext) -> dict | None:
    """Refunds over the cap go to a human (policy P10). Returning a dict skips the tool call."""
    if tool.name == "issue_refund":
        try:
            amount = float(args.get("amount", 0))
        except (TypeError, ValueError):
            return {"error": "amount must be a number"}
        if amount > config.REFUND_CAP_USD:
            tool_context.state["needs_human"] = True
            return {"error": f"Refunds over ${config.REFUND_CAP_USD:.0f} need a human agent (policy P10).",
                    "handoff": True}
    return None


def track_usage(callback_context: CallbackContext, llm_response: LlmResponse) -> None:
    """Counts tokens for every model call and stops the run when the eval budget is spent."""
    usage.record(llm_response.usage_metadata, model=llm_response.model_version)
    return None

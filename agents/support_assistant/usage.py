"""Token and cost tally for one process, with a hard cap so an eval run can't overspend.

Prices are USD per 1M tokens (Vertex AI, standard tier). Check current pricing before relying on them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

PRICES = {  # model: (input, output)
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-embedding-001": (0.15, 0.0),
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Tally:
    calls: int = 0
    by_model: dict[str, list[int]] = field(default_factory=dict)  # model -> [input, output]

    def usd(self) -> float:
        total = 0.0
        for model, (i, o) in self.by_model.items():
            pin, pout = price(model)
            total += i / 1e6 * pin + o / 1e6 * pout
        return total


def price(model: str) -> tuple[float, float]:
    """Longest matching prefix, so "gemini-2.5-flash-lite-001" finds "gemini-2.5-flash-lite"."""
    match = max((k for k in PRICES if model.startswith(k)), key=len, default=None)
    return PRICES[match] if match else PRICES["gemini-3.5-flash"]  # unknown model: assume the pricier one


TALLY = Tally()


def cap_usd() -> float:
    return float(os.getenv("EVAL_COST_CAP_USD", "1.00"))


def record(meta, model: str | None = None) -> None:
    from . import config

    if meta is None:
        return
    model = model or config.AGENT_MODEL
    io = TALLY.by_model.setdefault(model, [0, 0])
    io[0] += meta.prompt_token_count or 0
    io[1] += (meta.candidates_token_count or 0) + (getattr(meta, "thoughts_token_count", 0) or 0)
    TALLY.calls += 1
    if TALLY.usd() > cap_usd():
        raise BudgetExceeded(f"eval spend ${TALLY.usd():.4f} is over EVAL_COST_CAP_USD=${cap_usd():.2f}")

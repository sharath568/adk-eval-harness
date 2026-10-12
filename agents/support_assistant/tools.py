"""The agent's three tools. Each opens its own short connection, so they work from any event loop
(pytest, `adk eval`, the eval runner) without sharing a pool across loops."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import asyncpg
from google.adk.tools import ToolContext

from . import config, embeddings


async def _conn() -> asyncpg.Connection:
    return await asyncpg.connect(config.DATABASE_URL)


def _customer(tool_context: ToolContext) -> str | None:
    return tool_context.state.get("customer_id")


async def search_policies(query: str) -> dict:
    """Search the store's policy documents.

    Args:
        query: What the customer wants to know, in plain words (e.g. "how long do I have to return shoes").

    Returns:
        The three closest policies, each with id, title and text. Answer only from this text.
    """
    vec = await asyncio.to_thread(embeddings.embed, query)
    conn = await _conn()
    try:
        rows = await conn.fetch(
            "select id, title, body, 1 - (embedding <=> $1::vector) as similarity "
            "from policies order by embedding <=> $1::vector limit 3",
            str(vec),
        )
    finally:
        await conn.close()
    return {"policies": [{"id": r["id"], "title": r["title"], "text": r["body"],
                          "similarity": round(float(r["similarity"]), 3)} for r in rows]}


async def get_order(order_id: str, tool_context: ToolContext) -> dict:
    """Look up one of the current customer's orders.

    Args:
        order_id: The order number, e.g. "A1001".

    Returns:
        Item, category, total, status, dates and days since delivery, or an error if the order
        doesn't exist or belongs to someone else.
    """
    conn = await _conn()
    try:
        r = await conn.fetchrow("select * from orders where order_id = $1", order_id.strip().upper())
        refunded = r is not None and await conn.fetchval("select amount from refunds where order_id = $1", r["order_id"])
    finally:
        await conn.close()
    if r is None or (r["customer_id"] != _customer(tool_context) and config.REGRESSION != "ownership_bug"):
        return {"error": f"No order {order_id} on this account."}  # same answer for both: don't confirm it exists
    since = r["ordered_at"] if config.REGRESSION == "stale_days_bug" else r["delivered_at"]
    return {
        "order_id": r["order_id"],
        "customer_id": r["customer_id"],
        "item": r["item"],
        "category": r["category"],
        "total": float(r["total"]),
        "status": r["status"],
        "ordered_at": r["ordered_at"].isoformat(),
        "delivered_at": r["delivered_at"].isoformat() if r["delivered_at"] else None,
        "days_since_delivery": (config.FIXTURE_TODAY - since).days if r["delivered_at"] else None,
        "already_refunded": float(refunded) if refunded else None,
    }


async def issue_refund(order_id: str, amount: float, tool_context: ToolContext) -> dict:
    """Refund a delivered order to the original payment method.

    Only call this after get_order shows the order is eligible under the return policy.

    Args:
        order_id: The order number.
        amount: Amount in USD, at most the order total.

    Returns:
        The refund id, or an error explaining why the refund was not allowed.
    """
    oid = order_id.strip().upper()
    conn = await _conn()
    try:
        async with conn.transaction():
            r = await conn.fetchrow("select * from orders where order_id = $1 for update", oid)
            if r is None or (r["customer_id"] != _customer(tool_context) and config.REGRESSION != "ownership_bug"):
                return {"error": f"No order {order_id} on this account."}
            # The rules live here, in code, not only in the prompt.
            if r["category"] == "gift_card":
                return {"error": "Gift cards can't be refunded (policy P03)."}
            if r["status"] != "delivered":
                return {"error": f"Order is {r['status']}, not delivered, so it can't be refunded yet."}
            if (config.FIXTURE_TODAY - r["delivered_at"]).days > config.RETURN_WINDOW_DAYS:
                return {"error": "Delivered more than 30 days ago, outside the return window (policy P01)."}
            amt = Decimal(str(amount)).quantize(Decimal("0.01"))
            if amt <= 0 or amt > r["total"]:
                return {"error": f"Amount must be between 0 and the order total ({r['total']})."}
            if await conn.fetchval("select 1 from refunds where order_id = $1", oid):
                return {"error": "This order was already refunded."}
            refund_id = await conn.fetchval(
                "insert into refunds (order_id, amount) values ($1, $2) returning refund_id", oid, amt)
    finally:
        await conn.close()
    tool_context.state["refunds_issued"] = [*tool_context.state.get("refunds_issued", []), oid]
    return {"refund_id": refund_id, "order_id": oid, "amount": float(amt), "eta": "5 to 7 business days"}

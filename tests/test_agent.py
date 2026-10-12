"""Layer 1: wiring, tools, guardrails and state. Not intelligence: the model's moves are scripted."""

import pytest

from conftest import call, say, tool_results
from support_assistant import guardrails, regressions, usage
from support_assistant.agent import build_agent


async def test_policy_question_searches_and_model_sees_the_policy_text(harness, db):
    h = harness([call("search_policies", query="how many days do I have to return shoes"),
                 say("You have 30 days from delivery [P01].")])
    events = await h.send("How long do I have to return shoes?")
    hits = tool_results(events, "search_policies")[0]["policies"]
    assert hits[0]["id"] == "P01"
    # The second model request must carry the tool output, or the answer can't be grounded.
    seen = " ".join(c.model_dump_json() for c in h.llm.requests[1].contents)
    assert "30 days of delivery" in seen


async def test_customer_id_is_in_the_instruction(harness, db):
    h = harness([say("Hi!")], customer_id="C100")
    await h.send("hello")
    assert "customer C100" in h.llm.requests[0].config.system_instruction


async def test_get_order_counts_days_from_delivery(harness, db):
    h = harness([call("get_order", order_id="A1001"), say("ok")])
    order = tool_results(await h.send("status of A1001?"), "get_order")[0]
    assert order["status"] == "delivered"
    assert order["days_since_delivery"] == 16  # delivered 2026-09-15, fixture today 2026-10-01


async def test_another_customers_order_is_invisible(harness, db):
    h = harness([call("get_order", order_id="B2001"), say("not found")], customer_id="C100")
    assert tool_results(await h.send("what about B2001?"), "get_order")[0] == {"error": "No order B2001 on this account."}


async def test_refund_happy_path_writes_once_and_updates_state(harness, db):
    h = harness([call("get_order", order_id="A1001"), call("issue_refund", order_id="A1001", amount=89.5),
                 say("Refunded.")])
    events = await h.send("Please refund A1001")
    assert tool_results(events, "issue_refund")[0]["amount"] == 89.5
    assert await db.fetchval("select amount from refunds where order_id = 'A1001'") == 89.5
    assert (await h.state())["refunds_issued"] == ["A1001"]


@pytest.mark.parametrize("order_id,amount,why", [
    ("A1003", 50.0, "Gift cards"),
    ("A1002", 129.0, "outside the return window"),
    ("A1005", 74.25, "not delivered"),
    ("A1001", 500.0, "between 0 and the order total"),
    ("B2001", 310.0, "No order B2001"),
])
async def test_refund_rules_are_enforced_in_code(harness, db, order_id, amount, why):
    h = harness([call("issue_refund", order_id=order_id, amount=amount), say("no")])
    result = tool_results(await h.send(f"refund {order_id}"), "issue_refund")[0]
    assert why in result["error"]
    assert await db.fetchval("select count(*) from refunds") == 0


async def test_second_refund_on_same_order_is_refused(harness, db):
    h = harness([call("issue_refund", order_id="A1001", amount=89.5), say("done"),
                 call("issue_refund", order_id="A1001", amount=89.5), say("no")])
    await h.send("refund A1001")
    assert "already refunded" in tool_results(await h.send("refund A1001 again"), "issue_refund")[0]["error"]


async def test_refund_over_cap_is_stopped_before_the_tool_runs(harness, db):
    h = harness([call("issue_refund", order_id="A1004", amount=649.0), say("A human will contact you.")])
    result = tool_results(await h.send("refund my watch A1004"), "issue_refund")[0]
    assert result["handoff"] is True
    assert await db.fetchval("select count(*) from refunds") == 0
    assert (await h.state())["needs_human"] is True


def test_tool_descriptions_tell_the_model_when_to_use_them():
    # Tool docs are part of the prompt. A one-word description is a regression, not a style choice.
    for t in build_agent(model="unused").tools:
        assert len(t.description) > 40, f"{t.name} description is too thin: {t.description!r}"


def test_cost_cap_stops_the_run(monkeypatch):
    class Meta:
        prompt_token_count, candidates_token_count, thoughts_token_count = 2_000_000, 0, 0

    monkeypatch.setenv("EVAL_COST_CAP_USD", "0.10")
    monkeypatch.setattr(usage, "TALLY", usage.Tally())
    with pytest.raises(usage.BudgetExceeded):
        usage.record(Meta(), model="gemini-3.5-flash-lite")


def test_unknown_regression_name_fails_loudly():
    with pytest.raises(ValueError):
        regressions.apply("typo", rules={}, model="m")


def test_guardrail_rejects_non_numeric_amount():
    class T:
        name = "issue_refund"

    class Ctx:
        state = {}

    assert guardrails.refund_cap(T(), {"amount": "lots"}, Ctx()) == {"error": "amount must be a number"}

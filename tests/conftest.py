"""Layer 1 harness: a scripted model replaces Gemini, the real tools run against the seeded Postgres
fixture, and a fake embedder replaces Vertex AI. No network, no cost, a couple of seconds."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import AsyncGenerator

import pytest
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "agents"), str(ROOT / "tools"), str(ROOT)]

from support_assistant import embeddings  # noqa: E402
from support_assistant.agent import build_agent  # noqa: E402

import seed as seed_mod  # noqa: E402

VECTORS = json.loads((ROOT / "fixtures/policy_vectors.json").read_text())["vectors"]
TITLES = {"P01": "return window days", "P02": "refund timing card", "P03": "gift card",
          "P04": "damaged wrong item", "P05": "shipping times", "P06": "international shipping canada uk",
          "P07": "cancel order", "P08": "warranty electronics", "P09": "price match drop", "P10": "large refunds human"}


def fake_embed(text: str) -> list[float]:
    """Returns the stored vector of the policy whose keywords overlap the query most. Deterministic."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    best = max(TITLES, key=lambda p: (len(words & set(TITLES[p].split())), p == "P01"))
    return VECTORS[best]


class ScriptedLlm(BaseLlm):
    """Replays model turns in order and records every request, so tests can see what the model saw."""

    script: list = []
    requests: list = []

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        if not self.script:
            raise AssertionError("model called more times than scripted")
        yield LlmResponse(content=types.Content(role="model", parts=[self.script.pop(0)]))


def call(name: str, **args) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


def say(text: str) -> types.Part:
    return types.Part(text=text)


class Harness:
    def __init__(self, script: list[types.Part], customer_id: str = "C100"):
        self.llm = ScriptedLlm(model="scripted", script=list(script), requests=[])
        self.agent = build_agent(model=self.llm)
        self.sessions = InMemorySessionService()
        self.runner = Runner(app_name="support", agent=self.agent, session_service=self.sessions)
        self.customer_id = customer_id
        self.session_id: str | None = None

    async def send(self, text: str) -> list:
        if self.session_id is None:
            s = await self.sessions.create_session(app_name="support", user_id="u1",
                                                   state={"customer_id": self.customer_id})
            self.session_id = s.id
        msg = types.Content(role="user", parts=[types.Part(text=text)])
        return [e async for e in self.runner.run_async(user_id="u1", session_id=self.session_id, new_message=msg)]

    async def state(self) -> dict:
        s = await self.sessions.get_session(app_name="support", user_id="u1", session_id=self.session_id)
        return dict(s.state)


def tool_results(events, name: str) -> list[dict]:
    return [p.function_response.response for e in events if e.content for p in e.content.parts
            if p.function_response and p.function_response.name == name]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(embeddings, "embed", fake_embed)


@pytest.fixture
async def db():
    """Fresh fixture data for each test that touches the database."""
    url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/support")
    await seed_mod.seed(url)
    import asyncpg

    conn = await asyncpg.connect(url)
    yield conn
    await conn.close()


@pytest.fixture
def harness():
    return Harness

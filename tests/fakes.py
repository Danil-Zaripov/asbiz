# -*- coding: utf-8 -*-
"""Подставная модель. Тесты не ходят в сеть и не зависят от доступности шлюза"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Union

from desk.llm import Reply, ToolCall, Usage

Step = Union[str, Dict[str, Any], Callable[[List[dict]], Any]]


def text_reply(text: str, usage: Usage = None) -> Reply:
    return Reply(text, [], usage or Usage(calls=1), "end_turn", [{"type": "text", "text": text}])


class FakeLLM:
    """Подставная модель, которая отвечает по заранее заданному сценарию

    Шаг сценария: текст ответа, вызов инструмента {"tool": ..., "args": ...},
    несколько вызовов {"tools": [...]} или функция от истории сообщений
    """

    def __init__(self, script: List[Step], tokens: int = 100):
        self.script = list(script)
        self.calls: List[Dict[str, Any]] = []
        self.ledger: List[Usage] = []
        self.tokens = tokens

    def chat(self, messages: List[dict], **kw: Any) -> Reply:
        self.calls.append({"messages": [dict(m) for m in messages], **kw})
        if not self.script:
            raise AssertionError("сценарий подставной модели закончился")
        step = self.script.pop(0)
        if callable(step):
            step = step(messages)
        out = self.tokens // 4
        usage = Usage(self.tokens, out, 0, 0, 0.01, 0.0001, self.tokens + out, calls=1)
        self.ledger.append(usage)
        if isinstance(step, dict):
            steps = step["tools"] if "tools" in step else [step]
            calls = [ToolCall("toolu_%d_%d" % (len(self.calls), i), s["tool"], s["args"])
                     for i, s in enumerate(steps)]
            blocks = [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls]
            return Reply("", calls, usage, "tool_use", blocks)
        return text_reply(str(step), usage)

    async def achat(self, messages: List[dict], **kw: Any) -> Reply:
        return self.chat(messages, **kw)

    def total(self) -> Usage:
        out = Usage()
        for u in self.ledger:
            out = out + u
        return out

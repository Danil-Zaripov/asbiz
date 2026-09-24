# -*- coding: utf-8 -*-
"""Семинар 1. Потоковая выдача: разбор событий и время до первого токена. Сеть не нужна"""
import asyncio

import httpx
import pytest

from desk.config import Settings
from desk.llm import LLM, LLMError, Usage
from desk.stream import collect, iter_sse

SSE = """event: message_start
data: {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 25, "output_tokens": 1}}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}

event: ping
data: {"type": "ping"}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "пла"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "тежи"}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}

event: message_stop
data: {"type": "message_stop"}

"""


def usage_from(u, latency, retries):
    return Usage(int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)), 0, 0, latency,
                 calls=1, retries=retries)


def test_iter_sse_basic():
    events = list(iter_sse(SSE.splitlines()))
    assert [name for name, _ in events][:3] == ["message_start", "content_block_start", "ping"]
    assert events[3][1]["delta"]["text"] == "пла" and len(events) == 8


def test_iter_sse_edge_cases():
    lines = [": комментарий", "event: x", "data: {\"a\":", "data: 1}", "", "data: {\"b\": 2}"]
    assert list(iter_sse(lines)) == [("x", {"a": 1}), ("message", {"b": 2})]   # склейка и хвост
    assert list(iter_sse(["event:y", "data:{\"c\":3}", ""])) == [("y", {"c": 3})]  # без пробела
    assert list(iter_sse(["data: {\"d\": 4}\r", "\r"])) == [("message", {"d": 4})]  # переводы \r\n


class Clock:
    """Часы, которые сдвигаются на 0,1 с при каждом обращении"""
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.1
        return self.t


def test_collect_text_usage_and_ttft():
    res = collect(iter_sse(SSE.splitlines()), Clock(), 0.0, usage_from)
    assert res.text == "платежи" and res.stop_reason == "end_turn"
    assert (res.usage.input_tokens, res.usage.output_tokens) == (25, 3)   # выход из message_delta
    assert res.ttft_s == pytest.approx(0.1) and res.total_s == pytest.approx(0.2)
    assert not res.truncated


def test_collect_counts_thinking_and_waits_for_text():
    lines = SSE.replace('"type": "text_delta", "text": "пла"',
                        '"type": "thinking_delta", "thinking": "рассуждаю"').splitlines()
    res = collect(iter_sse(lines), Clock(), 0.0, usage_from)
    assert res.text == "тежи" and res.thinking_chars == len("рассуждаю")


def test_collect_without_text_and_truncated():
    lines = [l for l in SSE.splitlines() if "text_delta" not in l]
    lines = [l.replace("end_turn", "max_tokens") for l in lines]
    res = collect(iter_sse(lines), Clock(), 0.0, usage_from)
    assert res.text == "" and res.ttft_s is None and res.truncated


def test_error_event_stops_collection():
    lines = ["event: error", 'data: {"type": "error", "error": {"type": "overloaded_error", "message": "занято"}}', ""]
    with pytest.raises(LLMError):
        collect(iter_sse(lines), Clock(), 0.0, usage_from)


def sse_response(request):
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=SSE.encode("utf-8"))


def test_stream_through_client_and_retry_before_first_byte():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(529) if len(seen) == 1 else sse_response(request)

    sleeps = []
    llm = LLM(Settings("https://gw.example", "key"), transport=httpx.MockTransport(handler),
              sleep=sleeps.append, cache=False)
    res = llm.stream([{"role": "user", "content": "x"}], max_tokens=16)
    import json
    assert json.loads(seen[-1].content)["stream"] is True
    assert res.text == "платежи" and res.usage.retries == 1 and len(sleeps) == 1
    assert llm.total().calls == 1 and res.ttft_s is not None


def test_async_stream():
    async def handler(request):
        return sse_response(request)

    llm = LLM(Settings("https://gw.example", "key"), async_transport=httpx.MockTransport(handler), cache=False)
    res = asyncio.run(llm.astream([{"role": "user", "content": "x"}]))
    assert res.text == "платежи" and 0 <= res.ttft_s <= res.total_s

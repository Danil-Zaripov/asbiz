# -*- coding: utf-8 -*-
"""Потоковая выдача

С "stream": true шлюз присылает ответ по кусочкам, событиями. Событие
состоит из строк event и data, между событиями пустая строка:

    event: message_start
    data: {"type": "message_start", "message": {"usage": {"input_tokens": 25}}}

    event: content_block_delta
    data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Ок"}}

    event: message_delta
    data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}

    event: message_stop
    data: {"type": "message_stop"}

Здесь поток разбирается на события и собирается в ответ
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple

from .llm import LLMError, Usage


@dataclass
class StreamResult:
    """Ответ, собранный из потока"""

    text: str
    stop_reason: str
    usage: Usage
    ttft_s: Optional[float]  # до первого слова; None, если текста нет
    total_s: float  # до конца потока
    thinking_chars: int = 0  # длина рассуждения в символах

    @property
    def truncated(self) -> bool:
        """Ответ оборван по max_tokens"""
        return self.stop_reason == "max_tokens"


def iter_sse(lines: Iterable[str]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Разбирает строки потока на события: имя и данные

    Событие кончается пустой строкой. Несколько строк data одного события
    склеиваются. Строка с двоеточием в начале это комментарий
    """
    event: Optional[str] = None
    data = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if data:
                yield event or "message", json.loads("\n".join(data))
            event, data = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield event or "message", json.loads("\n".join(data))


def collect(
    events: Iterable[Tuple[str, Dict[str, Any]]],
    clock: Callable[[], float],
    started: float,
    usage_from: Callable[[Dict[str, Any], float, int], Usage],
    retries: int = 0,
) -> StreamResult:
    """Собирает ответ из событий

    Текст склеивается из кусочков, время до первого слова засекается
    на первом кусочке текста. Событие error обрывает сбор
    """
    usage_raw: Dict[str, Any] = {}
    parts, ttft, stop, thinking = [], None, "end_turn", 0
    for name, data in events:
        kind = data.get("type", name)
        if kind == "message_start":
            usage_raw.update((data.get("message") or {}).get("usage") or {})
        elif kind == "content_block_delta":
            delta = data.get("delta") or {}
            if delta.get("type") == "text_delta":
                if ttft is None:
                    ttft = clock() - started
                parts.append(delta.get("text", ""))
            elif delta.get("type") == "thinking_delta":
                thinking += len(delta.get("thinking", ""))
        elif kind == "message_delta":
            stop = (data.get("delta") or {}).get("stop_reason") or stop
            usage_raw.update(data.get("usage") or {})
        elif kind == "error":
            err = data.get("error") or {}
            raise LLMError(
                "поток прерван: %s %s" % (err.get("type"), err.get("message", ""))
            )
    total = clock() - started
    usage = usage_from(usage_raw, total, retries)
    return StreamResult("".join(parts).strip(), stop, usage, ttft, total, thinking)

# -*- coding: utf-8 -*-
"""Кэш префикса у провайдера

    python 3_prefix_cache.py

Длинная постановка уходит три раза с датой в начале и три раза с датой
в конце. Провайдер берёт из кэша совпадающее начало запроса. Смотрите,
в каком варианте это срабатывает
"""
from __future__ import annotations

import datetime
import time

from desk.config import DATA
from desk.llm import LLM


def system_variant(rules: str, stamp: str, where: str) -> str:
    """Постановка с меткой времени в начале или в конце"""
    if where == "start":
        return stamp + "\n\n" + rules
    if where == "end":
        return rules + "\n\n" + stamp
    raise ValueError("where: start или end")


def main() -> None:
    rules = (DATA / "razmetka.md").read_text(encoding="utf-8")
    llm = LLM(cache=False)
    question = (
        "Отнеси обращение к категории, ответь одним словом: "
        "«Не могу войти, пароль не подходит»"
    )
    print("| где метка | попытка | входных без кэша | из кэша | ответ |")
    print("|---|---|---|---|---|")
    for where in ("start", "end"):
        for attempt in range(1, 4):
            stamp = "Текущее время: %s" % datetime.datetime.now().isoformat(
                timespec="seconds"
            )
            reply = llm.chat(
                [
                    {"role": "system", "content": system_variant(rules, stamp, where)},
                    {"role": "user", "content": question},
                ],
                max_tokens=10,
            )
            u = reply.usage
            print(
                "| %s | %d | %d | %d | %s |"
                % (where, attempt, u.input_tokens, u.cache_read_tokens, reply.text)
            )
            time.sleep(1.1)  # чтобы метка времени сменилась
    total = llm.total()
    print("\nпотрачено взвешенных токенов: %.0f" % total.weighted)


if __name__ == "__main__":
    main()

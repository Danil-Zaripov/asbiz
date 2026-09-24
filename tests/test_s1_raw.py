# -*- coding: utf-8 -*-
"""Семинар 1, раунд 1. Запрос вручную: заголовки, тело, отправка, текст ответа"""
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from desk.config import Settings

ROOT = Path(__file__).resolve().parent.parent
FOUND = [p for p in (ROOT / "1_raw_request.py", ROOT / "seminars" / "s1" / "1_raw_request.py") if p.exists()]
if not FOUND:  # скрипты раунда 1 есть только в папке первого семинара и в эталоне
    pytest.skip("скрипты раунда 1 лежат в папке семинара 1", allow_module_level=True)
PATH = FOUND[0]
spec = importlib.util.spec_from_file_location("raw_request", PATH)
raw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw)

CFG = Settings("https://gw.example", "key-1", "deepseek-flash", extra_body={})


def test_headers_for_both_schemes():
    h = raw.headers(CFG)
    assert h["x-api-key"] == "key-1" and h["anthropic-version"] == "2023-06-01"
    assert "authorization" not in h
    hb = raw.headers(Settings("https://gw.example", "key-1", auth_scheme="bearer"))
    assert hb["authorization"] == "Bearer key-1" and "x-api-key" not in hb


def test_body_puts_system_in_its_own_field_and_keeps_extra():
    b = raw.body(CFG, raw.user("привет"), system="Будь краток", max_tokens=5, temperature=1.0)
    assert b == {"model": "deepseek-flash", "max_tokens": 5, "temperature": 1.0,
                 "messages": [{"role": "user", "content": "привет"}], "system": "Будь краток"}
    assert "system" not in raw.body(CFG, raw.user("привет"))
    with_default = raw.body(Settings("https://gw.example", "k"), raw.user("привет"))
    assert with_default["thinking"] == {"type": "disabled"}          # добавка из настроек
    on = raw.body(CFG, raw.user("привет"), extra={"thinking": {"type": "enabled"}})
    assert on["thinking"] == {"type": "enabled"}                     # extra сильнее настроек


def test_send_and_text_of():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"content": [{"type": "thinking", "thinking": "…"},
                                                     {"type": "text", "text": " плате"},
                                                     {"type": "text", "text": "жи "}],
                                         "stop_reason": "end_turn", "usage": {"input_tokens": 5}})

    resp = raw.send(CFG, raw.body(CFG, raw.user("x")), client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert str(seen[0].url) == "https://gw.example/v1/messages"
    assert seen[0].headers["x-api-key"] == "key-1" and json.loads(seen[0].content)["model"] == "deepseek-flash"
    assert raw.text_of(resp) == "платежи"


def test_send_raises_with_status():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad key")))
    with pytest.raises(RuntimeError) as e:
        raw.send(CFG, raw.body(CFG, raw.user("x")), client=client)
    assert "401" in str(e.value)


def test_prefix_cache_variants():
    spec = importlib.util.spec_from_file_location(
        "prefix_cache", next(p for p in (ROOT / "3_prefix_cache.py", ROOT / "seminars" / "s1" / "3_prefix_cache.py")
                             if p.exists()))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.system_variant("ПРАВИЛА", "метка", "start") == "метка\n\nПРАВИЛА"
    assert mod.system_variant("ПРАВИЛА", "метка", "end") == "ПРАВИЛА\n\nметка"
    with pytest.raises(ValueError):
        mod.system_variant("ПРАВИЛА", "метка", "middle")

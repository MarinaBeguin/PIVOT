"""Tests of the spending limits and the visitors' sessions of a public deployment."""

from pathlib import Path

import pytest

from pivot.meta.client import Usage
from pivot.scene.examples import office_scene
from pivot.server.limits import Budget, SpendLedger, chat_cost, visitor_key
from pivot.server.visitors import Visitors


def test_requests_are_priced_from_tokens() -> None:
    usage = Usage(calls=8, prompt_tokens=45_617, completion_tokens=3_679)
    assert chat_cost("muse-spark-1.3", usage) == pytest.approx(0.0727, abs=1e-4)
    # An unknown model is priced as the most expensive known one.
    assert chat_cost("muse-new", usage) == chat_cost("muse-wonder-1.0", usage)


def test_budget_reads_the_environment() -> None:
    budget = Budget.from_env({"PIVOT_BUDGET_USD": "12.5", "PIVOT_VISITOR_DAILY_REQUESTS": "3"})
    assert budget == Budget(12.5, Budget().visitor_daily_usd, 3)


def test_a_visitor_is_stopped_by_requests_or_dollars_per_day() -> None:
    ledger = SpendLedger(Budget(total_usd=10.0, visitor_daily_usd=0.5, visitor_daily_requests=2))
    ledger.record("a", 0.1)
    assert ledger.refusal("a") is None
    ledger.record("a", 0.1)
    assert "2 requests today" in (ledger.refusal("a") or "")
    ledger.record("b", 0.6, request=False)
    assert "share" in (ledger.refusal("b") or "")
    assert ledger.refusal("c") is None


def test_the_total_budget_stops_everyone_and_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    budget = Budget(total_usd=1.0, visitor_daily_usd=5.0, visitor_daily_requests=99)
    SpendLedger(budget, path).record("a", 1.2)
    restarted = SpendLedger(budget, path)
    assert restarted.total == pytest.approx(1.2)
    assert "whole budget" in (restarted.refusal("someone else") or "")


def test_daily_limits_start_over_the_next_day() -> None:
    day = ["2026-11-04"]
    ledger = SpendLedger(Budget(visitor_daily_requests=1), today=lambda: day[0])
    ledger.record("a", 0.0)
    assert ledger.refusal("a") is not None
    day[0] = "2026-11-05"
    assert ledger.refusal("a") is None


def test_visitor_key_trusts_only_the_proxy_entry() -> None:
    headers = {"x-forwarded-for": "6.6.6.6, 1.2.3.4"}
    forged = {"x-forwarded-for": "9.9.9.9, 1.2.3.4"}
    assert visitor_key(headers, "10.0.0.1", "salt") == visitor_key(forged, "10.0.0.1", "salt")
    assert visitor_key(headers, "10.0.0.1", "salt") != visitor_key(headers, "10.0.0.1", "other")
    assert visitor_key({}, "1.2.3.4", "salt") == visitor_key(headers, None, "salt")
    # Without trusted proxies, the header is ignored.
    assert visitor_key(headers, "5.5.5.5", "salt", proxy_hops=0) == \
        visitor_key({}, "5.5.5.5", "salt")
    # IPv6 addresses of one household's /64 count as one visitor.
    assert visitor_key({}, "2001:db8::1", "s") == visitor_key({}, "2001:db8::ffff", "s")
    assert len(visitor_key({}, None, "s")) == 32


def test_visitors_get_their_own_session_until_idle() -> None:
    now = [0.0]
    visitors = Visitors(office_scene(), max_visitors=2, idle_seconds=100.0, clock=lambda: now[0])
    token_a, a = visitors.get(None)
    token_b, b = visitors.get("unknown")
    assert token_a != token_b and a.session is not b.session
    assert visitors.get(token_a)[1] is a
    now[0] = 50.0
    visitors.get(token_a)
    token_c, _ = visitors.get(None)  # Over the limit: b, the least recently seen, goes.
    assert len(visitors) == 2 and visitors.get(token_c)[0] == token_c
    assert visitors.get(token_b)[0] != token_b
    now[0] = 500.0  # Everyone idle: a known token gets a fresh session.
    assert visitors.get(token_a)[0] != token_a


def test_reset_gives_a_fresh_session() -> None:
    visitors = Visitors(office_scene())
    token, visitor = visitors.get(None)
    assert visitors.reset(token).session is not visitor.session
    assert visitors.get(token)[1].session is not visitor.session

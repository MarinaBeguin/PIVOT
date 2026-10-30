"""Spending limits of a public deployment: what each request costs, and who may still spend.

Every request in words or by voice is priced from what the Meta API bills
(tokens, seconds of audio) and recorded in a ledger kept on disk, so that a
restarted server keeps counting. A request is refused once the deployment has
spent its total budget, or once a visitor has spent their daily share or made
their daily number of requests. Visitors are told apart by their IP address,
kept only as a keyed hash.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import ipaddress
import json
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from pivot.meta.client import Usage

# Dollars per million tokens (input, output), from dev.meta.ai pricing (Oct 30, 2026).
# Cached input is cheaper; it is priced as plain input, which errs on the safe side.
TOKEN_PRICES = {"muse-spark-1.3": (1.25, 4.25), "muse-wonder-1.0": (4.00, 20.00)}
# Muse Voice Transcribe: $0.18 per hour of audio.
TRANSCRIBE_PER_SECOND = 0.18 / 3600


def chat_cost(model: str, usage: Usage) -> float:
    """Prices the model calls of a request.

    Args:
        model: Model name; an unknown one is priced as the most expensive known.
        usage: Tokens spent.

    Returns:
        The cost, in dollars.
    """
    prompt, completion = TOKEN_PRICES.get(model, max(TOKEN_PRICES.values()))
    return (usage.prompt_tokens * prompt + usage.completion_tokens * completion) / 1e6


@dataclass(frozen=True)
class Budget:
    """Spending limits of a deployment.

    Attributes:
        total_usd: Most the deployment may ever spend.
        visitor_daily_usd: Most one visitor may spend per day (UTC).
        visitor_daily_requests: Most requests one visitor may make per day.
    """

    total_usd: float = 25.0
    visitor_daily_usd: float = 1.0
    visitor_daily_requests: int = 20

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Budget:
        """Reads the limits from ``PIVOT_BUDGET_USD``, ``PIVOT_VISITOR_DAILY_USD`` and
        ``PIVOT_VISITOR_DAILY_REQUESTS``, keeping the defaults for those not set.

        Args:
            environ: Environment variables; the process's if None.

        Returns:
            The budget.
        """
        environ = os.environ if environ is None else environ
        default = cls()
        return cls(
            float(environ.get("PIVOT_BUDGET_USD", default.total_usd)),
            float(environ.get("PIVOT_VISITOR_DAILY_USD", default.visitor_daily_usd)),
            int(environ.get("PIVOT_VISITOR_DAILY_REQUESTS", default.visitor_daily_requests)))


def _today() -> str:
    """Returns today's date in UTC, as ``YYYY-MM-DD``."""
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


class SpendLedger:
    """What a deployment and each visitor have spent, kept on disk if a path is given.

    Args:
        budget: The limits.
        path: JSON file holding the ledger; in memory only if None.
        today: Gives the current day, injectable for tests.
    """

    def __init__(self, budget: Budget, path: Path | None = None,
                 today: Callable[[], str] = _today):
        self.budget = budget
        self._path = path
        self._today = today
        self._lock = threading.Lock()
        self._total = 0.0
        # Day -> visitor -> [dollars, requests]; only today's entry is kept.
        self._days: dict[str, dict[str, list[float]]] = {}
        if path is not None and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._total = float(data["total"])
            self._days = data["days"]

    @property
    def total(self) -> float:
        """Dollars spent since the deployment started."""
        return self._total

    def _visitor_today(self, visitor: str) -> list[float]:
        """Returns a visitor's spending today, dropping earlier days.

        Args:
            visitor: Visitor key.

        Returns:
            ``[dollars, requests]``, a live entry of the ledger.
        """
        day = self._today()
        self._days = {day: self._days.get(day, {})}
        return self._days[day].setdefault(visitor, [0.0, 0])

    def refusal(self, visitor: str) -> str | None:
        """Says why a visitor may not make another request, if they may not.

        Args:
            visitor: Visitor key.

        Returns:
            The reason, written for the visitor, or None if the request may go on.
        """
        with self._lock:
            if self._total >= self.budget.total_usd:
                return ("The public demo has used its whole budget. You can run PIVOT yourself: "
                        "see the README.")
            dollars, requests = self._visitor_today(visitor)
        if requests >= self.budget.visitor_daily_requests:
            return (f"You have made {int(requests)} requests today, the most this public demo "
                    "allows per visitor. Come back tomorrow, or run PIVOT yourself.")
        if dollars >= self.budget.visitor_daily_usd:
            return ("You have used today's share of this public demo. Come back tomorrow, "
                    "or run PIVOT yourself.")
        return None

    def remaining(self, visitor: str) -> dict[str, float]:
        """Says how much a visitor may still use today, for the page to show.

        Args:
            visitor: Visitor key.

        Returns:
            Requests left today, and whether the deployment's budget is spent.
        """
        with self._lock:
            dollars, requests = self._visitor_today(visitor)
            spent = self._total >= self.budget.total_usd
        left = 0 if spent or dollars >= self.budget.visitor_daily_usd else \
            max(0, self.budget.visitor_daily_requests - int(requests))
        return {"requests_left": left, "budget_spent": spent}

    def record(self, visitor: str, dollars: float, request: bool = True) -> None:
        """Counts what a visitor spent, and saves the ledger.

        Args:
            visitor: Visitor key.
            dollars: The cost.
            request: Whether it counts as a request (a transcription only
                prepares the request that follows).
        """
        with self._lock:
            entry = self._visitor_today(visitor)
            entry[0] += dollars
            entry[1] += int(request)
            self._total += dollars
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                partial = self._path.with_suffix(".tmp")
                partial.write_text(json.dumps({"total": self._total, "days": self._days}),
                                   encoding="utf-8")
                partial.replace(self._path)


def visitor_key(headers: Mapping[str, str], peer: str | None, salt: str,
                proxy_hops: int = 1) -> str:
    """Identifies a visitor by a keyed hash of their IP address.

    Each trusted proxy in front of the server appends the address it saw to
    ``X-Forwarded-For``: the entry ``proxy_hops`` from the right is the one the
    nearest trusted proxy saw; entries further left come from the client and
    can be forged. IPv6 addresses count per /64 network (one household).

    Args:
        headers: Request headers.
        peer: Address of the socket's peer.
        salt: Secret key of the hash.
        proxy_hops: Number of trusted proxies; 0 ignores the header.

    Returns:
        32 hexadecimal characters; the address itself is not kept.
    """
    forwarded = [p.strip() for p in headers.get("x-forwarded-for", "").split(",") if p.strip()]
    address = forwarded[-proxy_hops] if 0 < proxy_hops <= len(forwarded) else (peer or "unknown")
    try:
        ip = ipaddress.ip_address(address)
        if ip.version == 6:
            address = str(ipaddress.ip_network(f"{ip}/64", strict=False))
    except ValueError:
        pass
    return hmac.new(salt.encode(), address.encode(), hashlib.sha256).hexdigest()[:32]


@dataclass(frozen=True)
class Meter:
    """Prices the requests of a public deployment and enforces its budget.

    Attributes:
        ledger: What was spent, and the limits.
        salt: Secret key of the visitors' address hashes.
        proxy_hops: Trusted proxies in front of the server (see :func:`visitor_key`).
        chat_price: Cost of the tokens of a request in words.
        audio_per_second: Cost of a second of transcribed audio.
    """

    ledger: SpendLedger
    salt: str
    proxy_hops: int = 1
    chat_price: Callable[[Usage], float] = lambda usage: 0.0
    audio_per_second: float = 0.0

    def visitor(self, headers: Mapping[str, str], peer: str | None) -> str:
        """Identifies the visitor of a request.

        Args:
            headers: Request headers.
            peer: Address of the socket's peer.

        Returns:
            The visitor key.
        """
        return visitor_key(headers, peer, self.salt, self.proxy_hops)

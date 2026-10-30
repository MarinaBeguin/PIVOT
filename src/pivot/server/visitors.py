"""One session per visitor: each visitor rearranges their own copy of the room.

A visitor is recognized by a random token kept in a cookie. Sessions idle for
too long are dropped, and the oldest ones too when there are many, so that a
public server's memory stays bounded.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pivot.agent.tools import Session
from pivot.scene.graph import Scene

COOKIE = "pivot_session"


@dataclass
class Visitor:
    """What the server keeps for one visitor.

    Attributes:
        session: Their copy of the scene and its history.
        agent: Their conversation with the agent, created on first use.
        last_seen: Time of their last request, from the store's clock.
    """

    session: Session
    agent: Any = None
    last_seen: float = field(default=0.0)


class Visitors:
    """The sessions of all visitors, keyed by their token.

    Args:
        scene: The scene every new session starts from.
        max_visitors: Most sessions kept; the least recently seen go first.
        idle_seconds: Sessions idle for longer are dropped.
        clock: Gives the time in seconds, injectable for tests.
    """

    def __init__(self, scene: Scene, max_visitors: int = 300, idle_seconds: float = 7200.0,
                 clock: Callable[[], float] = time.monotonic):
        self._scene = scene
        self._max = max_visitors
        self._idle = idle_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._visitors: dict[str, Visitor] = {}

    def __len__(self) -> int:
        return len(self._visitors)

    def get(self, token: str | None) -> tuple[str, Visitor]:
        """Returns a visitor's session, creating one for an unknown or missing token.

        Args:
            token: The token of the visitor's cookie, if any.

        Returns:
            The token (a new one for a new session) and the visitor.
        """
        now = self._clock()
        with self._lock:
            self._visitors = {t: v for t, v in self._visitors.items()
                              if now - v.last_seen <= self._idle}
            if token is None or token not in self._visitors:
                token = secrets.token_urlsafe(24)
                self._visitors[token] = Visitor(Session(self._scene), last_seen=now)
                while len(self._visitors) > self._max:
                    oldest = min((t for t in self._visitors if t != token),
                                 key=lambda t: self._visitors[t].last_seen)
                    del self._visitors[oldest]
            visitor = self._visitors[token]
            visitor.last_seen = now
            return token, visitor

    def reset(self, token: str) -> Visitor:
        """Restores a visitor's scene and starts a new conversation.

        Args:
            token: The visitor's token, as returned by :meth:`get`.

        Returns:
            The fresh visitor.
        """
        with self._lock:
            visitor = Visitor(Session(self._scene), last_seen=self._clock())
            self._visitors[token] = visitor
            return visitor

"""One session per visitor and per room: each visitor rearranges their own copies.

A visitor is recognized by a random token kept in a cookie. They get a session
for each room they open, and find it as they left it when they come back to
that room. Visitors idle for too long are dropped, and the least recently seen
too when there are many, so that a public server's memory stays bounded.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pivot.agent.tools import Session
from pivot.scene.graph import Scene

COOKIE = "pivot_session"


@dataclass
class Visitor:
    """What the server keeps for one visitor in one room.

    Attributes:
        session: Their copy of the scene and its history.
        agent: Their conversation with the agent, created on first use.
    """

    session: Session
    agent: Any = None


@dataclass
class _Entry:
    """A visitor's sessions, by room, and the time of their last request."""

    last_seen: float
    rooms: dict[str, Visitor] = field(default_factory=dict)


class Visitors:
    """The sessions of all visitors, keyed by their token and by room.

    Args:
        scenes: The scene every new session of a room starts from, by room id.
        max_visitors: Most visitors kept; the least recently seen go first.
        idle_seconds: Visitors idle for longer are dropped.
        clock: Gives the time in seconds, injectable for tests.
    """

    def __init__(self, scenes: Mapping[str, Scene], max_visitors: int = 300,
                 idle_seconds: float = 7200.0, clock: Callable[[], float] = time.monotonic):
        self._scenes = dict(scenes)
        self._max = max_visitors
        self._idle = idle_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, token: str | None, room: str) -> tuple[str, Visitor]:
        """Returns a visitor's session of a room, creating what is missing.

        Args:
            token: The token of the visitor's cookie, if any.
            room: The room's id.

        Returns:
            The token (a new one for a new visitor) and the visitor's session
            of that room.
        """
        now = self._clock()
        with self._lock:
            self._entries = {t: e for t, e in self._entries.items()
                             if now - e.last_seen <= self._idle}
            if token is None or token not in self._entries:
                token = secrets.token_urlsafe(24)
                self._entries[token] = _Entry(now)
                while len(self._entries) > self._max:
                    oldest = min((t for t in self._entries if t != token),
                                 key=lambda t: self._entries[t].last_seen)
                    del self._entries[oldest]
            entry = self._entries[token]
            entry.last_seen = now
            if room not in entry.rooms:
                entry.rooms[room] = Visitor(Session(self._scenes[room]))
            return token, entry.rooms[room]

    def reset(self, token: str, room: str) -> Visitor:
        """Restores a visitor's copy of a room and starts a new conversation.

        Args:
            token: The visitor's token, as returned by :meth:`get`.
            room: The room's id.

        Returns:
            The fresh session.
        """
        with self._lock:
            entry = self._entries.setdefault(token, _Entry(self._clock()))
            entry.rooms[room] = Visitor(Session(self._scenes[room]))
            return entry.rooms[room]

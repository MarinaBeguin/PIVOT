"""The PIVOT agent: a Muse model reasoning through geometric tools.

The loop is goal → tool call → geometric verdict → revised call, until the
model answers. The system prompt forbids claims that no tool result supports,
and every step is recorded so that answers can be audited and replayed.

Ask questions about a scene written by :mod:`pivot.build`::

    python -m pivot.agent --scene outputs/b20a261fdf/scene.json "Can chair 5 go under table 60?"
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from pivot.agent.tools import TOOLS, Session, call_tool
from pivot.meta.client import MetaConfig, Usage, call_with_retry
from pivot.scene.graph import Scene

# Sent when a request uses up its steps, so that the user still gets an answer.
_WRAP_UP = ("You have reached the step limit for this request. Without calling any tool, "
            "summarize in plain words what you did (all verified) and what remains to do.")

SYSTEM_PROMPT = """You are PIVOT, an assistant that rearranges a real room reconstructed in 3D.
Coordinates are meters in a z-up frame; the floor is z = 0. Yaw is in degrees.

Rules:
- Start with list_objects. Refer to objects by id.
- Never state that something fits, collides or can be moved unless a tool result in this
  conversation says so. Quote the numbers (gaps, clearances, effort) from the tool results.
- A free destination (check_placement, propose_placements) is not enough: use plan_motion
  or move_object to verify the object can actually get there.
- To make a move the user asked for, call move_object directly: it searches and verifies
  the path itself and moves nothing if there is none. Calling plan_motion first plans the
  same path twice. Use plan_motion only to test without moving, and do not repeat a check
  whose result you already have: these searches take seconds each.
- If a request is infeasible, say why with the measured numbers, then look for the closest
  feasible alternative (another spot, another orientation, moving another object first).
  Propose it, but do not move anything to an alternative the user has not agreed to.
- A result with goal_adjusted is NOT the pose asked for: before moving, check that the
  adjusted pose still does what the user asked (where, how far in, which side). If it does
  not, the request is infeasible: say so and propose the adjusted pose, without moving.
- To push an object under or against another (a chair under a table), first plan_motion
  aimed at the other object's center, with snap_m enough to reach its edge (half its depth
  plus half the object's, often 0.7 to 0.9): the nearest feasible pose is as far in as it
  goes, and aimed_at gives the share of the object that ends up under the other.
  If that share is below what the user asked (0 means it only stands against it, e.g. a
  chair taller than the space under the table top), it does not fit under: say so with the
  heights, and offer to push it against the table instead.
- A warning "stands in front of the door" means the passage is blocked; beside the door,
  against the same wall, raises no warning. A warning does not make a pose infeasible: if
  the user asked for that exact spot, do it and mention the warning. When several poses do
  what the user asked (along a wall, behind a desk), take one that keeps doors and windows
  clear; if none does, say so before anything else in your answer.
- Only call move_object when the user asked to move something, and remove_object when they
  asked to take something out (to get rid of it, or to replace it: the old sofa goes before
  the new one comes in). To try a new object (one to buy, or from another room of the
  user's), add_object with its size; a size estimated from a photo can be corrected with
  resize_object. When the user says where it goes, give that spot (x, y): new furniture is
  often delivered flat-packed and assembled in place, so no path from the door is needed.
  If objects stand there, name them and offer to move them aside (make_room); say from
  "doors" whether it would also pass the doors assembled. Then move it like any other object
  (with a verified path); furniture measured in the scan is never placed without a path.
- For a request with many moves (rearranging the room), proceed object by object and keep
  going until it is done. Then check with check_access that the room can still be walked
  through (or crossed in a wheelchair, width_m 0.9, if the user cares about it), and say
  so; if a move blocks a passage, say which and propose a fix. Never say the doors are
  clear unless doors_clear is true: name each piece of furniture in front of a door, and
  whether your moves put it there (there_before_any_move false) or it already stood there.
- To make room for someone at a place (a wheelchair at a table, access to a bed), call
  make_way once with that object: it chooses the spot and the passage that move the fewest
  objects and moves them. Do not move objects one by one for it. Say which door the way
  starts from, its width, and what was moved; if it fails, say why with its message.
- When an object needs two people to move (people_needed in list_objects), say so.
- When the home has several rooms (rooms in list_objects), call places by their room names.
  To move something into another room, use plan_to_room or move_to_room: they find a free
  spot there and a path through the doorways. Say which doors it goes through and the margin
  left in the tightest one. If it is too wide for a doorway, say so with both widths; if
  objects are in the way, offer to move them aside (make_room_and_move with the given spot).

How to answer: write for someone who is not an engineer. Plain sentences, distances in
centimeters, positions described by what is around them ("against the left wall, next to
the door"); coordinates only in parentheses when useful. Never mention tool names, field
names or statuses ("plan_motion", "invalid_goal", "min_clearance"): say what they mean
("I checked that it can be carried there without hitting anything; the tightest spot
leaves 7 cm").
- Answer in the user's language, briefly."""


# Languages of the viewer, by code, as the agent is told to answer in them.
LANGUAGE_NAMES = {"en": "English", "fr": "French", "de": "German", "es": "Spanish",
                  "it": "Italian", "zh": "Simplified Chinese"}


@dataclass
class Step:
    """One tool call of an agent run.

    Attributes:
        tool: Tool name.
        arguments: Decoded arguments.
        result: Decoded result.
    """

    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class AgentRun:
    """Outcome of one user request.

    Attributes:
        answer: The model's final answer.
        steps: Tool calls, in order.
        finished: False if the step budget ran out before an answer.
    """

    answer: str
    steps: list[Step] = field(default_factory=list)
    finished: bool = True


class Agent:
    """Runs user requests against a session with a Muse model.

    Args:
        session: The scene to work on; modified by ``move_object`` calls.
        config: API configuration; ``agent_model`` selects the model.
        client: SDK client; created from ``config`` if None.
        max_steps: Maximum number of model calls per request.
    """

    def __init__(self, session: Session, config: MetaConfig, client: OpenAI | None = None,
                 max_steps: int = 12):
        self.session = session
        self._config = config
        self._client = client if client is not None else config.client()
        self._max_steps = max_steps
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.usage = Usage()
        self._language: str | None = None

    def _complete(self, tools: bool = True) -> Any:
        """Requests the next model message.

        Args:
            tools: Whether the model may call tools; if not, it must answer.

        Returns:
            The SDK's first choice message.
        """
        # Messages and tools are plain dicts in the API's wire format, which the
        # SDK's TypedDict overloads cannot match statically.
        create = self._client.chat.completions.create
        effort = ({"reasoning_effort": self._config.reasoning_effort}
                  if self._config.reasoning_effort else {})
        response = call_with_retry(lambda: create(  # type: ignore[call-overload]
            model=self._config.agent_model,
            messages=self.messages,
            tools=[tool.schema() for tool in TOOLS],
            tool_choice="auto" if tools else "none",
            **effort,
        ))
        self.usage.add(response)
        return response.choices[0].message

    def ask(self, request: str, language: str | None = None,
            on_step: Callable[[str, dict[str, Any]], None] | None = None) -> AgentRun:
        """Handles one user request, keeping the conversation history.

        Args:
            request: The user's message.
            language: Code of the language of the user's page (see
                :data:`LANGUAGE_NAMES`); the answer is in it unless the user
                writes in another one.
            on_step: Told each tool call (name, arguments) before it runs, so
                that a long request can show what it is doing.

        Returns:
            The answer and the tool calls that support it.
        """
        name = LANGUAGE_NAMES.get(language or "")
        if name and language != self._language:
            self._language = language
            self.messages.append({"role": "system", "content": (
                f"The user's page is in {name}. Answer in {name} unless the user writes in "
                f"another language, then in theirs; name objects and rooms in that language.")})
        self.messages.append({"role": "user", "content": request})
        run = AgentRun(answer="")
        for _ in range(self._max_steps):
            message = self._complete()
            calls = message.tool_calls or []
            self.messages.append({
                "role": "assistant", "content": message.content or "",
                **({"tool_calls": [c.model_dump() for c in calls]} if calls else {}),
            })
            if not calls:
                run.answer = message.content or ""
                return run
            for call in calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {"raw": call.function.arguments}
                if on_step is not None:
                    on_step(call.function.name, arguments)
                result = call_tool(self.session, call.function.name, call.function.arguments)
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                run.steps.append(Step(call.function.name, arguments, json.loads(result)))
        run.finished = False
        run.answer = self._wrap_up()
        return run

    def _wrap_up(self) -> str:
        """Asks for a plain summary once the step budget is spent.

        Returns:
            What was done and what remains, in words.
        """
        self.messages.append({"role": "user", "content": _WRAP_UP})
        try:
            message = self._complete(tools=False)
        except Exception:  # pylint: disable=broad-exception-caught
            # The summary is a courtesy: without it, the user still gets the moves done.
            return "I ran out of steps before finishing; the moves done so far were verified."
        self.messages.append({"role": "assistant", "content": message.content or ""})
        return message.content or "I ran out of steps before finishing."


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: answers questions in one conversation.

    Args:
        argv: Command-line arguments; ``sys.argv`` if None.
    """
    parser = argparse.ArgumentParser(description="Asks the PIVOT agent about a scene.")
    parser.add_argument("--scene", type=Path, required=True, help="scene.json")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("questions", nargs="+")
    args = parser.parse_args(argv)
    agent = Agent(Session(Scene.load(args.scene)), MetaConfig.from_env(args.env_file))
    for question in args.questions:
        print(f"> {question}")
        run = agent.ask(question)
        for step in run.steps:
            print(f"  [{step.tool}] {json.dumps(step.arguments)}")
        print(run.answer if run.finished else f"(unfinished) {run.answer}", flush=True)

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
- If a request is infeasible, say why with the measured numbers, then look for the closest
  feasible alternative (another spot, another orientation, moving another object first).
  Propose it, but do not move anything to an alternative the user has not agreed to.
- To push an object under or against another (a chair under a table), aim at the other
  object's center with snap_m around 0.5: the nearest feasible pose is as far in as it goes.
- A warning "stands in front of the door" means the passage is blocked; beside the door,
  against the same wall, raises no warning. A warning does not make a pose infeasible: if
  the user asked for that exact spot, do it and mention the warning; avoid it only when the
  user asked to keep doors or windows clear.
- Only call move_object when the user asked to move something.
- For a request with many moves (rearranging the room), proceed object by object and keep
  going until it is done. Then check with check_access that the room can still be walked
  through (or crossed in a wheelchair, width_m 0.9, if the user cares about it), and say
  so; if a move blocks a passage, say which and propose a fix.
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

    def ask(self, request: str) -> AgentRun:
        """Handles one user request, keeping the conversation history.

        Args:
            request: The user's message.

        Returns:
            The answer and the tool calls that support it.
        """
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
                result = call_tool(self.session, call.function.name, call.function.arguments)
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {"raw": call.function.arguments}
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

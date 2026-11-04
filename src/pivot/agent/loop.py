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
import threading
from collections.abc import Callable
from functools import partial
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from openai import OpenAI

from pivot.agent.tools import TOOLS, Session, call_tool
from pivot.meta.client import DEFAULT_BASE_URL, MetaConfig, Usage, call_with_retry
from pivot.meta.products import look_up_size
from pivot.planning.motion import Stopped, stoppable
from pivot.planning.way import way_is_clear
from pivot.scene.graph import Scene

# Sent when a request uses up its steps, so that the user still gets an answer.
_WRAP_UP = ("You have reached the step limit for this request. Without calling any tool, "
            "explain what remains unconfirmed. All provisional edits for this request will be "
            "discarded: never say they were applied or tell the user to undo them.")

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
  If the user asked for a move and movable objects obstruct that same requested route,
  use make_room_and_move to plan and verify the full rearrangement before execution.
  Propose a different destination, but do not move to it without the user's agreement.
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
  against the same wall, raises no warning. Do not execute a move that newly blocks a door.
  When several poses do
  what the user asked (along a wall, behind a desk), take one that keeps doors and windows
  clear; if none does, say so before anything else in your answer.
- Only call move_object when the user asked to move something, and remove_object when they
  asked to take something out (to get rid of it, or to replace it: the old sofa goes before
  the new one comes in). To try a new object (one to buy, or from another room of the
  user's), add_object with its size; a size estimated from a photo can be corrected with
  resize_object. When the user names a product (brand and model), or gives a link to its page
  (a shop, a Marketplace ad: you can read it with look_up_product's url, never say you
  cannot open a link before trying), first look_up_product for its size and say where it
  comes from; a size from a photo is an estimate. When
  the user says where it goes, give that spot (x, y): new furniture is
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

SYSTEM_PROMPT += """
- Tool edits are prepared provisionally. A stopped or budget-exhausted request discards
  its new edits. A cancelled-plan note overrides earlier provisional move results.
- Describe moves using left, right, forward and back, never compass directions. Anchor
  these directions to an identified object or explicitly to the top-down moving plan;
  do not assume the user faces the same way as the camera.
- On failure, name only blockers and gaps actually measured by tools. If the tool did
  not measure a bottleneck, say the search did not find a route; do not invent a gap.
- Every wheelchair-access claim must be supported by a successful make_way result in the
  current request, after the last furniture move. Individual collision-free furniture
  moves do not prove wheelchair access. Do not ask whether to verify after claiming success.
- A request asking for the easiest table or route is advisory: compare options without
  moving furniture, and ask for agreement before rearranging for a proposed table.
- Keep usable furniture groups together: do not leave pupils' chairs without their table
  or park furniture in the requested passage. If the tools cannot confirm these constraints,
  explain the limitation instead of announcing success.
- Before sending the final answer, remove all compass directions (including door/wall
  descriptions). Use object identifiers and explicit top-down plan anchors instead.
- If wheelchair access to the requested table fails, check other suitable tables and
  propose a verified alternative. Do not move furniture for another table without consent.
- Use suggest_access for the easiest table or a comparison: it does not change the scene.
  A failed make_way already returns bounded alternative searches; describe these results
  instead of inferring usable places from check_access's arm-reach report.
- For facing an identified board or other object, set facing_object_id. If its identity
  is unknown, ask; never claim an orientation the geometric tools have not checked.
"""


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


# The result of a tool call the user stopped, and the note left in the conversation.
_STOPPED = json.dumps({"error": "Stopped by the user."})
_STOPPED_NOTE = "(The user stopped this request; its provisional plan was cancelled.)"


@dataclass
class AgentRun:
    """Outcome of one user request.

    Attributes:
        answer: The model's final answer.
        steps: Tool calls, in order.
        finished: False if the step budget ran out before an answer.
        stopped: True if the user stopped the request before an answer.
    """

    answer: str
    steps: list[Step] = field(default_factory=list)
    finished: bool = True
    stopped: bool = False


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
        if config.base_url == DEFAULT_BASE_URL and session.look_up is None:
            # Web search is a Meta API tool; a local open-weight server has none.
            session.look_up = partial(look_up_size, config, client=self._client)
        self._max_steps = max_steps
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.usage = Usage()
        self._language: str | None = None
        self._request_lock = threading.Lock()

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
            on_step: Callable[[str, dict[str, Any]], None] | None = None,
            should_stop: Callable[[], bool] | None = None) -> AgentRun:
        """Serializes requests and prepares their edits before application.

        Args:
            request: User request.
            language: User interface language.
            on_step: Progress callback.
            should_stop: Cancellation callback.

        Returns:
            Answer and recorded planning steps.

        Raises:
            RuntimeError: If another request is already being prepared.
        """
        if self._request_lock.locked():
            raise RuntimeError("Another request is already being prepared.")
        with self._request_lock:
            return self._prepare(request, language, on_step, should_stop)

    def _prepare(self, request: str, language: str | None = None,
                 on_step: Callable[[str, dict[str, Any]], None] | None = None,
                 should_stop: Callable[[], bool] | None = None) -> AgentRun:
        """Prepares a request in an isolated session before applying its changes.

        Args:
            request: User request.
            language: User interface language.
            on_step: Progress callback for planned tool calls.
            should_stop: Cancellation callback.

        Returns:
            Recorded answer and steps. Stopped or budget-exhausted plans do
            not modify the original session.
        """
        original = self.session
        before_scene, spent_before = original.scene, original.spent
        message_count, previous_language = len(self.messages), self._language
        trial = replace(original, history=list(original.history), motions=list(original.motions),
                        motion_counts=list(original.motion_counts))
        self.session = trial
        try:
            run = self._ask(request, language, on_step, should_stop)
        except Exception:
            self.messages = self.messages[:message_count]
            self._language = previous_language
            raise
        finally:
            self.session = original
            # Bill actual external tool work even if its geometric plan is cancelled.
            original.spent += trial.spent - spent_before
        if trial.scene is not before_scene and not self._edits_succeeded(run):
            run.finished = False
        if run.finished and not run.stopped and (not all(
                way_is_clear(trial.scene, target, width, (depth, breadth), facing)
                for target, (width, depth, breadth, facing) in trial.access_goals.items())
                or not self._access_goals_hold(run, trial.scene)):
            run.finished = False
            run.answer = self._fallback(run)
            self.messages.append({"role": "assistant", "content": run.answer})
        if run.finished and not run.stopped and original.scene is before_scene:
            self._apply_trial(original, trial)
        elif trial.scene is not original.scene:
            run.finished = False
            for step in run.steps:
                step.result["applied"] = False
                if step.result.get("moved"):
                    step.result["moved"] = False
            notes = {
                "fr": "Le plan n'est pas terminé : aucun nouveau déplacement n'a été appliqué.",
                "de": "Der Plan ist unvollständig: Es wurden keine neuen Bewegungen angewendet.",
                "es": "El plan está incompleto: no se ha aplicado ningún movimiento nuevo.",
                "it": "Il piano è incompleto: nessun nuovo spostamento è stato applicato.",
                "zh": "计划尚未完成：未执行任何新的移动。",
            }
            run.answer = notes.get(self._language or "", "The plan is incomplete: no new "
                                   "moves have been applied.")
            measured = self._measured_failure(run)
            if measured:
                run.answer += " " + measured
            self.messages.append({"role": "assistant", "content": run.answer})
        return run

    @staticmethod
    def _access_goals_hold(run: AgentRun, scene: Scene) -> bool:
        """Rechecks successful access goals after every proposed edit is complete.

        Args:
            run: Tool results claiming that access was made possible.
            scene: Final proposed scene.

        Returns:
            True if all claimed access goals still hold in the final layout.
        """
        for step in run.steps:
            if step.tool != "make_way" or not step.result.get("success") \
                    or not step.arguments.get("execute", True):
                continue
            target = int(step.arguments["object_id"])
            if target not in scene.objects or not way_is_clear(
                    scene, target, float(step.arguments.get("width_m", .9)),
                    (float(step.arguments.get("spot_depth_m", 1.)),
                     float(step.arguments.get("spot_width_m", .8))),
                    step.arguments.get("facing_object_id")):
                return False
        return True

    @staticmethod
    def _edits_succeeded(run: AgentRun) -> bool:
        """Checks the last mutating tool outcome before applying a multi-step plan.

        Args:
            run: Recorded provisional tool outcomes.

        Returns:
            False if the last attempted edit failed; a later successful retry
            can supersede an earlier failure. This does not judge natural-language goals.
        """
        flags = {"move_object": "moved", "move_to_room": "moved",
                 "make_room_and_move": "moved", "make_way": "success",
                 "add_object": "added", "remove_object": "removed",
                 "resize_object": "resized", "undo": "undone"}
        for step in reversed(run.steps):
            if step.tool in flags:
                return bool(step.result.get(flags[step.tool]))
        return True

    @staticmethod
    def _apply_trial(original: Session, trial: Session) -> None:
        """Applies a completed plan, grouping its new edits into one undo action.

        Args:
            original: Unchanged live session.
            trial: Prepared session, possibly including undo requests.
        """
        extends_history = (len(trial.history) > len(original.history)
                           and all(left is right for left, right in
                                   zip(original.history, trial.history)))
        extends_motions = (len(trial.motions) >= len(original.motions)
                           and all(left is right for left, right in
                                   zip(original.motions, trial.motions)))
        if trial.scene is not original.scene and extends_history and extends_motions:
            original.record(trial.scene, trial.motions[len(original.motions):])
        else:
            # Undo/reset requests deliberately replace existing history.
            original.scene = trial.scene
            original.history = trial.history
            original.motions = trial.motions
            original.motion_counts = trial.motion_counts
        original.access_goals = dict(trial.access_goals)

    def _ask(self, request: str, language: str | None = None,
             on_step: Callable[[str, dict[str, Any]], None] | None = None,
             should_stop: Callable[[], bool] | None = None) -> AgentRun:
        """Handles one user request, keeping the conversation history.

        Args:
            request: The user's message.
            language: Code of the language of the user's page (see
                :data:`LANGUAGE_NAMES`); the answer is in it unless the user
                writes in another one.
            on_step: Told each tool call (name, arguments) before it runs, so
                that a long request can show what it is doing.
            should_stop: Tells whether the user asked to stop: looked at between
                steps, and inside the motion planner (see :mod:`pivot.planning.cancel`).

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
        stop = should_stop or (lambda: False)
        with stoppable(stop):
            for _ in range(self._max_steps):
                if stop():
                    return self._stopped(run)
                message = self._complete()
                calls = message.tool_calls or []
                self.messages.append({
                    "role": "assistant", "content": message.content or "",
                    **({"tool_calls": [c.model_dump() for c in calls]} if calls else {}),
                })
                if not calls:
                    run.answer = (message.content or "").strip()
                    if not run.answer:
                        run.finished = False
                        run.answer = self._wrap_up().strip() or self._fallback(run)
                        self.messages.append({"role": "assistant", "content": run.answer})
                    return run
                for call in calls:
                    self._run_call(call, run, on_step, stop)
                if run.stopped:
                    return self._stopped(run)
        run.finished = False
        run.answer = self._wrap_up().strip() or self._fallback(run)
        return run

    def _fallback(self, run: AgentRun) -> str:
        """Reports an incomplete request without inventing its geometric outcome.

        Args:
            run: Recorded tool results of the incomplete request.

        Returns:
            Localized summary stating that provisional changes were not applied.
        """
        texts = {
            "fr": ("Je n'ai pas pu confirmer la demande complète. Aucun nouveau changement "
                   "n'a été appliqué pour cette demande ; la scène reste dans son état précédent."),
            "de": ("Ich konnte die gesamte Anfrage nicht bestätigen. Für diese Anfrage wurde "
                   "keine neue Änderung angewendet; die Szene bleibt im vorherigen Zustand."),
            "es": ("No he podido confirmar toda la solicitud. No se ha aplicado ningún cambio "
                   "nuevo para esta solicitud; la escena conserva su estado anterior."),
            "it": ("Non ho potuto confermare l'intera richiesta. Non è stata applicata alcuna "
                   "nuova modifica per questa richiesta; la scena resta nello stato precedente."),
            "zh": "未能确认整个请求已完成。本次请求未应用任何新更改；场景保持原先状态。",
        }
        text = texts.get(self._language or "", "I could not confirm the complete request. "
                         "No new changes were applied for this request; the scene remains "
                         "in its previous state.")
        measured = self._measured_failure(run)
        return text + (" " + measured if measured else "")

    def _measured_failure(self, run: AgentRun) -> str:
        """Words a measured candidate gap without claiming all routes are impossible.

        Args:
            run: Tool results, possibly including candidate bottlenecks.

        Returns:
            Localized measured gap, or an empty string when no measurement exists.
        """
        templates = {
            "en": ("On the attempted passage between {names}: "
                   "{gap} cm available, {need} cm required."),
            "fr": ("Sur le passage essayé entre {names} : "
                   "{gap} cm disponibles, {need} cm nécessaires."),
            "de": ("Am versuchten Durchgang zwischen {names}: "
                   "{gap} cm frei, {need} cm erforderlich."),
            "es": ("En el paso intentado entre {names}: "
                   "{gap} cm disponibles, se necesitan {need} cm."),
            "it": ("Nel passaggio provato tra {names}: "
                   "{gap} cm disponibili, {need} cm necessari."),
            "zh": "尝试的通道位于 {names} 之间：可用宽度 {gap} 厘米，需要 {need} 厘米。",
        }
        for step in reversed(run.steps):
            gaps = step.result.get("candidate_bottlenecks", [])
            if not gaps:
                continue
            gap = gaps[0]
            names = " / ".join(f"{obj['label']} (#{obj['id']})" for obj in gap["objects"])
            return templates.get(self._language or "", templates["en"]).format(
                names=names, gap=round(100 * gap["gap_m"]),
                need=round(100 * gap["required_width_m"]))
        return ""

    def _run_call(self, call: Any, run: AgentRun,
                  on_step: Callable[[str, dict[str, Any]], None] | None,
                  stop: Callable[[], bool]) -> None:
        """Runs one tool call of the model and records its result.

        Once the user asked to stop, the remaining calls are answered "stopped"
        without running: the conversation keeps a result for every call.

        Args:
            call: The model's tool call.
            run: The request so far; its steps get this one.
            on_step: Told the call before it runs.
            stop: Tells whether the user asked to stop.
        """
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {"raw": call.function.arguments}
        result = _STOPPED
        if not run.stopped and not stop():
            if on_step is not None:
                on_step(call.function.name, arguments)
            try:
                result = call_tool(self.session, call.function.name, call.function.arguments)
            except Stopped:
                result = _STOPPED
        run.stopped = run.stopped or result == _STOPPED
        self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        if result != _STOPPED:
            run.steps.append(Step(call.function.name, arguments, json.loads(result)))

    def _stopped(self, run: AgentRun) -> AgentRun:
        """Ends a request the user stopped.

        Args:
            run: The request so far.

        Returns:
            It, marked stopped, with a local summary of recorded moves.
        """
        run.stopped, run.finished = True, False
        run.answer = self._fallback(run)
        self.messages.append({"role": "assistant", "content": _STOPPED_NOTE})
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
            return ""
        self.messages.append({"role": "assistant", "content": message.content or ""})
        return (message.content or "").strip()


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

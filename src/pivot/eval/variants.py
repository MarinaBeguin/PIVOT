"""The four ways of answering a task that the evaluation compares.

- ``llm_only``: the model sees the object list and answers with a pose, unchecked.
- ``planner_only``: no language understanding; the geometric search tries the
  free spots next to the anchor (or the requested point) until a motion works.
- ``llm_check``: the model proposes a pose, a motion plan checks it, and the
  model only hears "infeasible" until it gives up or the budget runs out.
- ``pivot``: the full agent, reasoning with every geometric tool.

The last two get the same budget of geometric checks. Every final pose is then
judged in the same way: reachable from the start (exact motion planning), and
satisfying the task's predicate. A refusal is right only for infeasible tasks.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from openai import OpenAI

from pivot.agent.loop import Agent
from pivot.agent.tools import Session, list_objects, plan_path, suggest_placements
from pivot.eval.tasks import CARRY_HEIGHT, NEXT_TO_GAP, Task, TaskKind, satisfies, solve
from pivot.meta.client import MetaConfig, Usage, call_with_retry
from pivot.geometry.box import OrientedBox
from pivot.planning.motion import MotionPlan, straight_motion
from pivot.planning.placement import ConflictKind, check_placement
from pivot.scene.graph import Scene

# Tools that run a geometric check.
CHECK_TOOLS = {"check_placement", "propose_placements", "plan_motion", "move_object",
               "make_room", "make_room_and_move"}

POSE_PROMPT = """You rearrange a real room reconstructed in 3D. Coordinates are meters in a
z-up frame; the floor is z = 0; yaw is in degrees. Objects, with their oriented boxes:
{scene}

Answer the request with a JSON object only, no other text:
{{"action": "move", "x": <center x>, "y": <center y>, "yaw_deg": <yaw>}} to move the object
there, or {{"action": "refuse", "reason": "<why>"}} if it cannot be done."""


@dataclass
class Outcome:
    """What a variant did for one task.

    Attributes:
        pose: Final ``(x, y, yaw_deg)`` of the object, None for a refusal.
        checks: Geometric checks used.
        seconds: Wall-clock time.
        motions: Motions the variant executed (other objects may have been
            moved aside, the object carried onto a table); empty if it only
            answered with a pose.
        usage: Model calls and tokens spent.
        note: Free-text detail (the model's answer, an error).
    """

    pose: tuple[float, float, float] | None
    checks: int = 0
    seconds: float = 0.0
    motions: list[MotionPlan] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    note: str = ""


@dataclass(frozen=True)
class Verdict:
    """How an outcome is judged.

    Attributes:
        success: Right refusal, or a reachable pose satisfying the task.
        claimed: Whether the variant answered with a pose.
        reachable: Whether that pose can be reached without collision.
        satisfies: Whether that pose is what the task asks for.
        collision_m: For an unreachable claimed pose, how deeply the object
            would collide there, in meters: the cost of believing the answer.
        detail: Why executed motions failed their replay, if they did.
    """

    success: bool
    claimed: bool
    reachable: bool = False
    satisfies: bool = False
    collision_m: float = 0.0
    detail: str = ""

    @property
    def false_claim(self) -> bool:
        """A pose was given but it is unreachable or not what was asked."""
        return self.claimed and not self.success


def judge(scene: Scene, task: Task, outcome: Outcome) -> Verdict:
    """Judges an outcome against the task, with exact geometry.

    Args:
        scene: The initial scene.
        task: The task.
        outcome: What the variant did.

    Returns:
        The verdict.
    """
    if outcome.pose is None:
        return Verdict(success=not task.feasible, claimed=False)
    if outcome.motions:
        # Executed moves are replayed and every segment is checked again.
        final = replay(scene, outcome.motions)
        if final is None:
            return Verdict(success=False, claimed=True,
                           detail=replay_failure(scene, outcome.motions))
        ok = satisfies(final, task, final[task.object_id])
        return Verdict(success=ok, claimed=True, reachable=True, satisfies=ok)
    end = reach(scene, task.object_id, outcome.pose)
    if end is None:
        return Verdict(success=False, claimed=True,
                       collision_m=_collision_depth(scene, task.object_id, outcome.pose))
    ok = satisfies(scene, task, dataclasses.replace(scene[task.object_id], box=end))
    return Verdict(success=ok, claimed=True, reachable=True, satisfies=ok)


def replay_failure(scene: Scene, motions: list[MotionPlan]) -> str:
    """Describes the first segment of executed motions that fails its check.

    Args:
        scene: The initial scene.
        motions: The motions, in execution order.

    Returns:
        The motion, the segment and the conflicts at its end; empty if none fails.
    """
    for index, motion in enumerate(motions):
        for segment, (a, b) in enumerate(zip(motion.waypoints, motion.waypoints[1:])):
            start = scene.moved(motion.object_id, a)
            if straight_motion(start, motion.object_id, b) is None:
                conflicts = check_placement(start, motion.object_id, b).conflicts
                where = "; ".join(c.describe() for c in conflicts) or "between its ends"
                return f"motion {index} (#{motion.object_id}), segment {segment}: {where}"
        scene = scene.moved(motion.object_id, motion.waypoints[-1])
    return ""


def reach(scene: Scene, object_id: int,
          pose: tuple[float, float, float]) -> OrientedBox | None:
    """Tests whether an object can be brought to a pose given in words.

    It is slid along the floor or, failing that, carried (and put down on
    whatever surface is under the pose).

    Args:
        scene: The scene.
        object_id: The object.
        pose: Requested ``(x, y, yaw_deg)``.

    Returns:
        The object's final box, or None if neither move is collision-free.
    """
    x, y, yaw = pose
    for carry in ({}, {"carry_height_m": CARRY_HEIGHT}):
        plan = plan_path(Session(scene), object_id=object_id, x=x, y=y, yaw_deg=yaw, **carry)
        if plan["status"] == "success":
            end_x, end_y, end_yaw, end_z = plan["waypoints"][-1]
            return scene[object_id].box.moved(np.array([end_x, end_y]), math.radians(end_yaw),
                                              end_z)
    return None


def _collision_depth(scene: Scene, object_id: int, pose: tuple[float, float, float]) -> float:
    """Measures how deeply an object would collide at a claimed pose, on the floor.

    Args:
        scene: The scene.
        object_id: The object.
        pose: Claimed ``(x, y, yaw_deg)``.

    Returns:
        The deepest collision, in meters; 0 if the pose itself is free (the
        object cannot get there, or the pose is outside the floor).
    """
    box = scene[object_id].box.moved(np.array(pose[:2]), math.radians(pose[2]))
    report = check_placement(scene, object_id, box)
    return max((c.amount for c in report.conflicts if c.kind is ConflictKind.COLLISION),
               default=0.0)


def replay(scene: Scene, motions: list[MotionPlan]) -> Scene | None:
    """Replays executed motions, checking every segment exactly.

    Args:
        scene: The initial scene.
        motions: The motions, in execution order.

    Returns:
        The final scene, or None if a segment is not collision-free.
    """
    for motion in motions:
        for a, b in zip(motion.waypoints, motion.waypoints[1:]):
            if straight_motion(scene.moved(motion.object_id, a), motion.object_id, b) is None:
                return None
        scene = scene.moved(motion.object_id, motion.waypoints[-1])
    return scene


def _pose_of(scene: Scene, object_id: int) -> tuple[float, float, float]:
    """Returns an object's pose.

    Args:
        scene: The scene.
        object_id: The object.

    Returns:
        ``(x, y, yaw_deg)``.
    """
    box = scene[object_id].box
    return float(box.center[0]), float(box.center[1]), math.degrees(box.yaw)


def parse_pose_answer(text: str) -> dict[str, Any]:
    """Reads the JSON answer of the pose-only variants.

    Args:
        text: The model's message.

    Returns:
        The decoded object; ``{"action": "refuse"}`` if none can be read.
    """
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        answer = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        answer = {}
    if answer.get("action") != "move" or not {"x", "y"} <= answer.keys():
        return {"action": "refuse", "reason": answer.get("reason", text)}
    return answer


class PoseModel:
    """A model asked for a pose in one message, given the object list.

    Args:
        scene: The scene described to the model.
        config: API configuration.
        client: SDK client; created from ``config`` if None.
    """

    def __init__(self, scene: Scene, config: MetaConfig, client: OpenAI | None = None):
        self._config = config
        self._client = client if client is not None else config.client()
        description = json.dumps(list_objects(Session(scene)))
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": POSE_PROMPT.format(scene=description)}]
        self.usage = Usage()

    def ask(self, content: str) -> dict[str, Any]:
        """Sends a message and reads the pose in the reply.

        Args:
            content: The user message.

        Returns:
            The decoded answer.
        """
        self.messages.append({"role": "user", "content": content})
        create = self._client.chat.completions.create
        # Messages are plain dicts in the API's wire format (see Agent._complete).
        response = call_with_retry(lambda: create(
            model=self._config.agent_model,
            messages=self.messages,  # type: ignore[arg-type]
        ))
        self.usage.add(response)
        message = response.choices[0].message
        self.messages.append({"role": "assistant", "content": message.content or ""})
        return parse_pose_answer(message.content or "")


def _pose_from(answer: dict[str, Any], scene: Scene, object_id: int) -> tuple[float, float, float]:
    """Turns a decoded answer into a pose, keeping the yaw if none is given.

    Args:
        answer: The decoded ``move`` answer.
        scene: The scene.
        object_id: The object.

    Returns:
        ``(x, y, yaw_deg)``.
    """
    yaw = answer.get("yaw_deg")
    return (float(answer["x"]), float(answer["y"]),
            float(yaw) if yaw is not None else _pose_of(scene, object_id)[2])


def llm_only(scene: Scene, task: Task, config: MetaConfig,
             client: OpenAI | None = None) -> Outcome:
    """The model answers alone, from the object list.

    Args:
        scene: The initial scene.
        task: The task.
        config: API configuration.
        client: SDK client.

    Returns:
        The outcome.
    """
    model = PoseModel(scene, config, client)
    answer = model.ask(task.prompt)
    if answer["action"] != "move":
        return Outcome(None, usage=model.usage, note=str(answer.get("reason", "")))
    return Outcome(_pose_from(answer, scene, task.object_id), usage=model.usage)


def llm_check(scene: Scene, task: Task, config: MetaConfig, client: OpenAI | None = None,
              budget: int = 8) -> Outcome:
    """The model proposes poses; each is checked, and only pass or fail is reported back.

    Args:
        scene: The initial scene.
        task: The task.
        config: API configuration.
        client: SDK client.
        budget: Maximum number of checks.

    Returns:
        The outcome: the first pose that passed, or a refusal.
    """
    model = PoseModel(scene, config, client)
    answer = model.ask(task.prompt)
    for check in range(1, budget + 1):
        if answer["action"] != "move":
            return Outcome(None, checks=check - 1, usage=model.usage,
                           note=str(answer.get("reason", "")))
        pose = _pose_from(answer, scene, task.object_id)
        if reach(scene, task.object_id, pose) is not None:
            return Outcome(pose, checks=check, usage=model.usage)
        if check < budget:
            answer = model.ask("That pose is infeasible. Propose another one, or refuse.")
    return Outcome(None, checks=budget, usage=model.usage, note="Budget exhausted.")


def planner_only(scene: Scene, task: Task, budget: int = 8) -> Outcome:
    """The geometric search alone: free spots next to the anchor, or the requested point.

    It reads the task's structured fields, not its words: constraints stated
    only in words (keeping doors clear) are ignored. Pushing under, putting
    on and reaching a blocked spot use the same moves as the task labels.

    Args:
        scene: The initial scene.
        task: The task.
        budget: Maximum number of motion plans.

    Returns:
        The outcome: the first reachable candidate, or a refusal.
    """
    if task.kind in (TaskKind.TUCK_UNDER, TaskKind.ON_SURFACE, TaskKind.AT_POINT_BLOCKED):
        solved = solve(scene, task)
        if solved is None:
            return Outcome(None, checks=1)
        # One move, or a move then making room.
        return Outcome(_pose_of(solved.scene, task.object_id),
                       checks=1 if len(solved.motions) == 1 else 2,
                       motions=list(solved.motions))
    session = Session(scene)
    if task.kind is TaskKind.AT_POINT:
        assert task.point is not None
        candidates = [(*task.point, None)]
        checks = 0
    else:
        assert task.anchor_id is not None
        proposals = suggest_placements(session, task.object_id, task.anchor_id, NEXT_TO_GAP,
                                       max_results=budget)["proposals"]
        candidates = [(*p["center_xy"], p["yaw_deg"]) for p in proposals]
        checks = 1
    for x, y, yaw in candidates:
        if checks >= budget:
            break
        checks += 1
        plan = plan_path(session, object_id=task.object_id, x=x, y=y, yaw_deg=yaw)
        if plan["status"] == "success":
            return Outcome((x, y, yaw if yaw is not None else _pose_of(scene, task.object_id)[2]),
                           checks=checks)
    return Outcome(None, checks=checks)


def pivot(scene: Scene, task: Task, config: MetaConfig, client: OpenAI | None = None,
          budget: int = 8) -> Outcome:
    """The full agent, with every geometric tool.

    Args:
        scene: The initial scene.
        task: The task.
        config: API configuration.
        client: SDK client.
        budget: Check budget; the agent gets three times as many model
            turns, as some turns only list or measure objects.

    Returns:
        The outcome: where the object ended up, or a refusal if it did not move.
    """
    session = Session(scene)
    agent = Agent(session, config, client, max_steps=3 * budget)
    run = agent.ask(task.prompt)
    checks = sum(step.tool in CHECK_TOOLS for step in run.steps)
    if session.scene[task.object_id].box is scene[task.object_id].box:
        return Outcome(None, checks=checks, usage=agent.usage, note=run.answer)
    return Outcome(_pose_of(session.scene, task.object_id), checks=checks,
                   motions=list(session.motions), usage=agent.usage, note=run.answer)


def timed(run: Callable[[], Outcome]) -> Outcome:
    """Runs a variant and records its wall-clock time.

    Args:
        run: The variant, with its arguments bound.

    Returns:
        Its outcome, with ``seconds`` set.
    """
    start = time.perf_counter()
    outcome = run()
    outcome.seconds = time.perf_counter() - start
    return outcome

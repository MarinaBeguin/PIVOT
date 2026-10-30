"""Motion planning: can an object actually get from here to there?

A valid destination is not enough: a 90 cm wide sofa cannot pass through an
80 cm door whatever the free space behind it. ``plan_motion`` searches the
object's ``(x, y, yaw)`` configuration space with A*, then re-validates every
step of the path with the same exact geometry as
:func:`pivot.planning.placement.check_placement`. A step that fails exact
validation is excluded and the search is repeated, so a returned path is
always exactly collision-free.

Assumptions: the object slides and rotates about the vertical axis, it is
never tilted or lifted, and objects resting on it move rigidly with it.
"""

from __future__ import annotations

import dataclasses
import heapq
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from pivot.geometry.box import OrientedBox, wrap_angle
from pivot.geometry.collision import box_distance_lower_bound
from pivot.geometry.solid import solid_distance
from pivot.planning.cspace import ConfigurationSpace, State
from pivot.planning.placement import ConflictKind, PlacementReport, check_placement
from pivot.scene.graph import Scene, carried_objects, is_opening

_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
# Cost multiplier of steps that leave less than the comfort margin to obstacles.
_TIGHT_COST_FACTOR = 4.0


class MotionStatus(str, Enum):
    """Outcome of a motion query."""

    SUCCESS = "success"
    NO_PATH = "no_path"
    SEARCH_LIMIT_REACHED = "search_limit_reached"
    INVALID_START = "invalid_start"
    INVALID_GOAL = "invalid_goal"


@dataclass(frozen=True)
class MotionPlan:
    """Result of :func:`plan_motion`.

    Attributes:
        status: Outcome of the search.
        object_id: The moved object.
        waypoints: Poses from start to goal; consecutive poses are joined by
            a straight translation or an in-place rotation. Empty on failure.
        expansions: Number of A* expansions used.
        message: Human-readable explanation.
        min_clearance: Smallest distance to an obstacle along the path, in
            meters, if a path was found.
    """

    status: MotionStatus
    object_id: int
    waypoints: list[OrientedBox] = field(default_factory=list)
    expansions: int = 0
    message: str = ""
    min_clearance: float | None = None

    @property
    def length(self) -> float:
        """Total translation distance along the path, in meters."""
        return float(sum(np.linalg.norm(b.center[:2] - a.center[:2])
                         for a, b in zip(self.waypoints, self.waypoints[1:])))

    @property
    def rotation(self) -> float:
        """Total absolute rotation along the path, in radians."""
        return float(sum(abs(_yaw_step(a.yaw, b.yaw))
                         for a, b in zip(self.waypoints, self.waypoints[1:])))

    @property
    def effort(self) -> float:
        """Distance travelled by the object's corners, in meters.

        Translation plus rotation weighted by the half diagonal of the
        footprint: the arc a corner sweeps, a proxy for the work of carrying
        or pushing the object.
        """
        if not self.waypoints:
            return 0.0
        size = self.waypoints[0].size
        return self.length + 0.5 * float(np.hypot(size[0], size[1])) * self.rotation

    def to_dict(self) -> dict[str, Any]:
        """Serializes the plan for tool-call responses.

        Returns:
            A compact JSON-compatible dictionary.
        """
        result: dict[str, Any] = {
            "status": self.status.value,
            "object_id": self.object_id,
            "message": self.message,
            "expansions": self.expansions,
        }
        if self.waypoints:
            result.update({
                "path_length_m": round(self.length, 2),
                "total_rotation_deg": round(math.degrees(self.rotation)),
                "effort_m": round(self.effort, 2),
                "min_clearance_m": (round(self.min_clearance, 3)
                                    if self.min_clearance is not None else None),
                # x, y, yaw in degrees, and the height of the bottom (lifted moves).
                "waypoints": [[round(float(b.center[0]), 3), round(float(b.center[1]), 3),
                               round(math.degrees(b.yaw), 1), round(b.z_min, 3)]
                              for b in self.waypoints],
            })
        return result


def _yaw_step(yaw_a: float, yaw_b: float) -> float:
    """Returns the smallest rotation between two rectangle orientations.

    Rectangles are symmetric under a half turn, so the result lies in
    [-pi/2, pi/2].

    Args:
        yaw_a: Start yaw, in radians.
        yaw_b: End yaw, in radians.

    Returns:
        The signed rotation, in radians.
    """
    delta = wrap_angle(yaw_b - yaw_a)
    if delta > 0.5 * math.pi:
        delta -= math.pi
    elif delta < -0.5 * math.pi:
        delta += math.pi
    return delta


@dataclass(frozen=True)
class _Validator:
    """Exact collision checks of straight motions between poses.

    Attributes:
        scene: The scene.
        object_id: The moving object.
        clearance: Required distance to other objects, in meters.
        floor_margin: Margin of the on-floor test, see ``check_placement``.
        max_step: Maximum translation between two checked poses, in meters.
        max_turn: Maximum rotation between two checked poses, in radians.
        carried: Ids of the objects resting on the moving object.
    """

    scene: Scene
    object_id: int
    clearance: float
    floor_margin: float
    max_step: float = 0.02
    max_turn: float = math.radians(2.0)
    carried: tuple[int, ...] = ()

    @classmethod
    def create(cls, scene: Scene, object_id: int, clearance: float,
               floor_margin: float) -> _Validator:
        """Builds a validator, computing the carried objects once.

        Args:
            scene: The scene.
            object_id: The moving object.
            clearance: Required distance to other objects, in meters.
            floor_margin: Margin of the on-floor test, see ``check_placement``.

        Returns:
            The validator.
        """
        return cls(scene, object_id, clearance, floor_margin,
                   carried=tuple(carried_objects(scene, object_id)))

    def pose_ok(self, box: OrientedBox) -> bool:
        """Checks one pose exactly.

        Args:
            box: The object's box at that pose.

        Returns:
            True if the pose has no conflict.
        """
        return check_placement(self.scene, self.object_id, box, self.clearance,
                               self.floor_margin, carried=self.carried).feasible

    def segment_ok(self, start: OrientedBox, end: OrientedBox) -> bool:
        """Checks a straight translation combined with a rotation, exactly.

        Args:
            start: Box at the start of the motion.
            end: Box at its end.

        Returns:
            True if every interpolated pose is free.
        """
        turn = _yaw_step(start.yaw, end.yaw)
        distance = float(np.linalg.norm(end.center[:2] - start.center[:2]))
        steps = max(1, math.ceil(distance / self.max_step), math.ceil(abs(turn) / self.max_turn))
        poses = [start.moved((1 - t) * start.center[:2] + t * end.center[:2], start.yaw + t * turn)
                 for t in np.linspace(0.0, 1.0, steps + 1)[1:]]
        index = 0
        while index < len(poses):
            report = self._report(poses[index])
            if not report.feasible:
                # Voxel aliasing: two volumes sliding past each other can overlap
                # by one cell at a single intermediate pose. A real contact lasts.
                last = index == len(poses) - 1
                if last or not _aliasing(report) or not self.pose_ok(poses[index + 1]):
                    return False
                index += 1
            index += 1
        return True

    def _report(self, box: OrientedBox) -> PlacementReport:
        """Checks one pose exactly, with the details of its conflicts.

        Args:
            box: The object's box at that pose.

        Returns:
            The placement report.
        """
        return check_placement(self.scene, self.object_id, box, self.clearance,
                               self.floor_margin, carried=self.carried)


# Volumes are 4 cm cells: an isolated overlap up to this depth is a sampling artifact.
_ALIASING_DEPTH = 0.05


def _aliasing(report: PlacementReport) -> bool:
    """Tells whether a pose's conflicts could be voxel aliasing only.

    Args:
        report: The pose's placement report.

    Returns:
        True if every conflict is a collision at most one cell deep.
    """
    return all(c.kind is ConflictKind.COLLISION and c.amount <= _ALIASING_DEPTH
               for c in report.conflicts)


# How far, and with how many exact checks, a pose is linked to the lattice.
_CONNECT_RADIUS = 0.6
_CONNECT_CHECKS = 40


def _connect(cspace: ConfigurationSpace, validator: _Validator, box: OrientedBox,
             to_lattice: bool) -> tuple[State, list[OrientedBox]] | None:
    """Links an arbitrary pose to a nearby free lattice state.

    The lattice inflates obstacles, so an object tucked in a tight spot (a
    chair under a desk) can be far from any free state: states are tried up
    to ``_CONNECT_RADIUS`` away, nearest first, each link checked exactly.
    If no direct link works (the object can only slide straight out), it
    escapes along straight lines first, see :func:`_escape`.

    Args:
        cspace: The configuration space.
        validator: Exact checker.
        box: The off-lattice pose.
        to_lattice: Direction of the link: from ``box`` to the lattice if True.

    Returns:
        The linked state and the intermediate poses from ``box`` to it (empty
        for a direct link), or None if no nearby state can be reached.
    """
    state = _link(cspace, validator, box, to_lattice, _CONNECT_CHECKS)
    if state is not None:
        return state, []
    return _escape(cspace, validator, box, to_lattice)


def _link(cspace: ConfigurationSpace, validator: _Validator, box: OrientedBox,
          to_lattice: bool, checks: int) -> State | None:
    """Links a pose to one of the nearest free lattice states by a straight motion.

    Args:
        cspace: The configuration space.
        validator: Exact checker.
        box: The off-lattice pose.
        to_lattice: Direction of the link: from ``box`` to the lattice if True.
        checks: Number of nearest states tried.

    Returns:
        The linked state, or None.
    """
    row, col = cspace.nearest_cell(box.center[:2])
    reach = int(math.ceil(_CONNECT_RADIUS / cspace.floor.resolution))
    candidates = [(k, row + dr, col + dc) for k in range(cspace.yaw_bins)
                  for dr in range(-reach, reach + 1) for dc in range(-reach, reach + 1)]
    candidates = [s for s in candidates if cspace.is_free(s)]
    def distance(state: State) -> float:
        """Measures how far a lattice state is from ``box``.

        Args:
            state: Candidate lattice state.

        Returns:
            Translation offset plus rotation, weighted at 10 cm per radian.
        """
        offset = float(np.linalg.norm(cspace.position(state[1], state[2]) - box.center[:2]))
        return offset + 0.1 * abs(_yaw_step(box.yaw, cspace.yaw(state[0])))

    candidates.sort(key=distance)
    for state in candidates[:checks]:
        lattice_box = cspace.box_at(box, state)
        ok = (validator.segment_ok(box, lattice_box) if to_lattice
              else validator.segment_ok(lattice_box, box))
        if ok:
            return state
    return None


def _escape(cspace: ConfigurationSpace, validator: _Validator, box: OrientedBox,
            to_lattice: bool, directions: int = 16) -> tuple[State, list[OrientedBox]] | None:
    """Slides a pose straight out in several directions until it can be linked.

    A chair pushed under a desk can often only move straight back: any
    diagonal or turning link to the lattice touches the desk.

    Args:
        cspace: The configuration space.
        validator: Exact checker.
        box: The off-lattice pose.
        to_lattice: Direction of the link: from ``box`` to the lattice if True.
        directions: Number of directions tried, evenly spread.

    Returns:
        The linked state and the escape pose, or None.
    """
    step = cspace.floor.resolution
    for angle in np.linspace(0.0, 2.0 * math.pi, directions, endpoint=False):
        direction = np.array([math.cos(angle), math.sin(angle)])
        current = box
        for distance in np.arange(step, _CONNECT_RADIUS + 1e-9, step):
            moved = box.moved(box.center[:2] + distance * direction)
            if not validator.segment_ok(current, moved):
                break
            current = moved
            state = _link(cspace, validator, current, to_lattice, checks=4)
            if state is not None:
                return state, [current]
    return None


@dataclass
class _Search:
    """A* over the lattice, with a shared expansion budget.

    Attributes:
        cspace: The configuration space.
        turn_cost: Cost of one yaw bin, in meters (arc length swept by the
            footprint's corners).
        budget: Remaining expansions.
        expansions: Expansions used so far.
        tight: Transitions found to sweep closer than the comfort margin
            between their endpoints; they cost as much as tight states.
    """

    cspace: ConfigurationSpace
    turn_cost: float
    budget: int
    expansions: int = 0
    tight: set[tuple[State, State]] = field(default_factory=set)

    def neighbors(self, state: State) -> list[tuple[State, float]]:
        """Lists the free neighbors of a state with their step costs.

        Args:
            state: Current state.

        Returns:
            Pairs of neighbor state and transition cost.
        """
        k, row, col = state
        resolution = self.cspace.floor.resolution
        result = [((k, row + dr, col + dc), resolution * math.hypot(dr, dc))
                  for dr, dc in _NEIGHBORS]
        bins = self.cspace.yaw_bins
        result += [(((k + dk) % bins, row, col), self.turn_cost) for dk in (-1, 1)]
        # Tight steps cost more: people keep their distance when moving furniture.
        return [(s, c if self.cspace.roomy[s] and (state, s) not in self.tight
                 else _TIGHT_COST_FACTOR * c)
                for s, c in result if self.cspace.is_free(s)]

    def heuristic(self, state: State, goal: State) -> float:
        """Admissible cost-to-go estimate.

        Args:
            state: Current state.
            goal: Goal state.

        Returns:
            Straight-line distance plus the minimal number of turns' cost.
        """
        k, row, col = state
        gk, grow, gcol = goal
        bins = self.cspace.yaw_bins
        turns = min((k - gk) % bins, (gk - k) % bins)
        distance = self.cspace.floor.resolution * math.hypot(row - grow, col - gcol)
        return distance + self.turn_cost * turns

    def run(self, start: State, goal: State, excluded: set[tuple[State, State]]
            ) -> list[State] | None:
        """Finds a shortest lattice path.

        Args:
            start: Start state.
            goal: Goal state.
            excluded: Transitions known to fail exact validation.

        Returns:
            The path from start to goal, None if the reachable space was
            exhausted, or an empty list if the budget ran out.
        """
        frontier: list[tuple[float, float, State]] = [(self.heuristic(start, goal), 0.0, start)]
        parents: dict[State, State | None] = {start: None}
        costs = {start: 0.0}
        while frontier:
            if self.budget <= 0:
                return []
            _, cost, state = heapq.heappop(frontier)
            if cost > costs[state]:
                continue
            if state == goal:
                path = [state]
                while (parent := parents[path[-1]]) is not None:
                    path.append(parent)
                return path[::-1]
            self.budget -= 1
            self.expansions += 1
            for neighbor, step in self.neighbors(state):
                new_cost = cost + step
                if (state, neighbor) in excluded or new_cost >= costs.get(neighbor, math.inf):
                    continue
                costs[neighbor] = new_cost
                parents[neighbor] = state
                heapq.heappush(frontier, (new_cost + self.heuristic(neighbor, goal),
                                          new_cost, neighbor))
        return None


def _motion_signature(a: OrientedBox, b: OrientedBox) -> tuple[float, float, float]:
    """Summarizes a lattice step as its translation direction and turn sign.

    Args:
        a: Pose before the step.
        b: Pose after the step.

    Returns:
        ``(dx, dy, turn_sign)`` with a unit (or zero) translation direction.
    """
    delta = b.center[:2] - a.center[:2]
    norm = float(np.linalg.norm(delta))
    direction = delta / norm if norm > 1e-9 else np.zeros(2)
    return (round(float(direction[0]), 3), round(float(direction[1]), 3),
            float(np.sign(round(_yaw_step(a.yaw, b.yaw), 6))))


def _simplify(boxes: list[OrientedBox]) -> list[OrientedBox]:
    """Drops intermediate poses of straight runs and of consecutive turns.

    Merged turns stay below a quarter turn: beyond that, the rotation between
    two rectangle poses becomes ambiguous.

    Args:
        boxes: Dense path.

    Returns:
        The path's endpoints and turning points.
    """
    kept = [boxes[0]]
    for i in range(1, len(boxes) - 1):
        turn = abs(_yaw_step(kept[-1].yaw, boxes[i + 1].yaw))
        if (_motion_signature(boxes[i - 1], boxes[i]) != _motion_signature(boxes[i], boxes[i + 1])
                or turn >= 0.5 * math.pi - 1e-6):
            kept.append(boxes[i])
    if len(boxes) > 1:
        kept.append(boxes[-1])
    return kept


def _min_clearance(scene: Scene, object_id: int, boxes: list[OrientedBox]) -> float:
    """Computes the smallest distance to an obstacle or a wall along a path.

    Args:
        scene: The scene.
        object_id: The moving object.
        boxes: Poses along the path.

    Returns:
        The minimum distance, in meters; infinity if nothing is within 1 m.
    """
    ignored = {object_id, *carried_objects(scene, object_id)}
    shape = scene[object_id].shape
    obstacles = [o for o in scene.objects.values()
                 if o.object_id not in ignored and not is_opening(o)]
    horizon = 1.0  # Beyond 1 m, clearance is not resolved.
    nearest = horizon
    for box in boxes:
        for other in obstacles:
            # Only obstacles that may beat the current minimum are measured.
            if box_distance_lower_bound(box, other.box) < nearest:
                nearest = min(nearest, solid_distance(box, shape, other.box, other.shape,
                                                      horizon=nearest))
    if nearest >= horizon:
        nearest = math.inf
    if scene.floor is not None:
        nearest = min(nearest, *(scene.floor.distance_to_walls(b.footprint()) for b in boxes))
    return nearest


def densify(boxes: list[OrientedBox], step: float = 0.05,
             turn: float = math.radians(5.0)) -> list[OrientedBox]:
    """Interpolates poses along a path, for measurements between waypoints.

    Args:
        boxes: Waypoints.
        step: Maximum translation between poses, in meters.
        turn: Maximum rotation between poses, in radians.

    Returns:
        The interpolated poses, including the waypoints.
    """
    dense = [boxes[0]]
    for a, b in zip(boxes, boxes[1:]):
        rotation = _yaw_step(a.yaw, b.yaw)
        distance = float(np.linalg.norm(b.center[:2] - a.center[:2]))
        count = max(1, math.ceil(distance / step), math.ceil(abs(rotation) / turn))
        for t in np.linspace(0.0, 1.0, count + 1)[1:]:
            dense.append(a.moved((1 - t) * a.center[:2] + t * b.center[:2], a.yaw + t * rotation))
    return dense


def _shortcut(boxes: list[OrientedBox], validator: _Validator) -> list[OrientedBox]:
    """Replaces lattice staircases with straight motions where they are roomy.

    From each kept pose, jumps to the farthest later pose reachable by one
    straight motion that passes ``validator``. The validator should demand the
    comfort margin: a shortcut must not trade the planner's preference for
    roomy configurations against a shorter, tighter motion. Consecutive
    waypoints are kept as they are, since they were already validated.

    Args:
        boxes: Validated waypoints, start first.
        validator: Exact checker of shortcuts, with the comfort margin.

    Returns:
        A path with the same endpoints and fewer, longer segments.
    """
    smoothed = [boxes[0]]
    i = 0
    while i < len(boxes) - 1:
        j = len(boxes) - 1
        while j > i + 1 and not validator.segment_ok(boxes[i], boxes[j]):
            j -= 1
        smoothed.append(boxes[j])
        i = j
    return smoothed


def _success(scene: Scene, object_id: int, waypoints: list[OrientedBox],
             expansions: int) -> MotionPlan:
    """Wraps a validated path into a successful plan.

    Args:
        scene: The scene.
        object_id: The moved object.
        waypoints: Validated, smoothed waypoints.
        expansions: Search expansions used.

    Returns:
        The plan, with its minimum clearance measured along the whole path.
    """
    return MotionPlan(MotionStatus.SUCCESS, object_id, waypoints, expansions,
                      "A collision-free motion was found.",
                      _min_clearance(scene, object_id, densify(waypoints)))


@dataclass(frozen=True)
class MotionOptions:
    """Parameters of a motion query.

    Attributes:
        clearance: Required distance to other objects along the way, in meters.
        yaw_bins: Yaw resolution, as a number of bins over half a turn.
        max_expansions: Search budget, shared across repairs.
        max_repairs: Maximum number of re-searches after exact validation
            rejects a lattice transition.
        floor_margin: Margin of the on-floor test, see ``check_placement``.
        comfort: Extra margin, beyond ``clearance``, that the planner tries to
            keep: tighter steps cost more and shortcuts must respect it.
        refine_yaw_bins: Before concluding ``NO_PATH``, search again with this
            many yaw bins, so that a passage needing an intermediate angle is
            not missed. No refinement if not larger than ``yaw_bins``.
    """

    clearance: float = 0.0
    yaw_bins: int = 8
    max_expansions: int = 200_000
    max_repairs: int = 25
    floor_margin: float = 0.025
    comfort: float = 0.08
    refine_yaw_bins: int = 16


def _tight_sweeps(path: list[State], boxes: list[OrientedBox], cspace: ConfigurationSpace,
                  roomy: _Validator) -> set[tuple[State, State]]:
    """Finds steps between roomy states whose sweep is nevertheless tight.

    Args:
        path: Lattice states of a validated path.
        boxes: The corresponding boxes.
        cspace: The configuration space.
        roomy: Exact checker with the comfort margin.

    Returns:
        The tight transitions, as ``(from, to)`` state pairs.
    """
    return {(path[i], path[i + 1]) for i in range(len(path) - 1)
            if cspace.roomy[path[i]] and cspace.roomy[path[i + 1]]
            and not roomy.segment_ok(boxes[i], boxes[i + 1])}


@dataclass
class _Repairer:
    """Alternates lattice searches with exact validation of their paths.

    Two kinds of repairs follow a search. A transition failing exact
    validation is excluded. A valid transition whose sweep, between its
    lattice endpoints, comes closer than the comfort margin is penalized like
    a tight state. The best valid path so far is kept, so penalties can only
    improve the result.

    Attributes:
        cspace: The configuration space.
        search: The A* search, holding the budget and penalties.
        validator: Exact checker with the required clearance.
        roomy: Exact checker with the comfort margin.
    """

    cspace: ConfigurationSpace
    search: _Search
    validator: _Validator
    roomy: _Validator

    def run(self, start: tuple[OrientedBox, State], goal: tuple[OrientedBox, State],
            max_repairs: int) -> tuple[list[OrientedBox] | None, bool]:
        """Searches, validates and repairs until the path stops improving.

        Args:
            start: Start pose and its linked lattice state.
            goal: Goal pose and its linked lattice state.
            max_repairs: Maximum number of re-searches.

        Returns:
            A tuple ``(path, exhausted)``: the best valid path (None if none was
            found) and whether the last search exhausted the reachable space.
        """
        excluded: set[tuple[State, State]] = set()
        best: list[OrientedBox] | None = None
        for _ in range(max_repairs + 1):
            path = self.search.run(start[1], goal[1], excluded)
            if not path:
                return best, path is None
            boxes = [self.cspace.box_at(start[0], s) for s in path]
            failed = next((i for i in range(len(path) - 1)
                           if not self.validator.segment_ok(boxes[i], boxes[i + 1])), None)
            if failed is not None:
                excluded.update({(path[failed], path[failed + 1]),
                                 (path[failed + 1], path[failed])})
                continue
            # The links to the start and goal poses were validated by _connect.
            best = [start[0], *boxes, goal[0]]
            newly_tight = _tight_sweeps(path, boxes, self.cspace, self.roomy) - self.search.tight
            if not newly_tight:
                break
            self.search.tight.update(newly_tight | {(b, a) for a, b in newly_tight})
        return best, False


@dataclass(frozen=True)
class _Ends:
    """The start and goal of a search, linked to the lattice.

    Attributes:
        start: Pose linked to the lattice on the start side, and its state.
        goal: Pose linked to the lattice on the goal side, and its state.
        before: Poses preceding ``start`` (the actual start, after an escape).
        after: Poses following ``goal`` (the actual goal, after an escape).
    """

    start: tuple[OrientedBox, State]
    goal: tuple[OrientedBox, State]
    before: list[OrientedBox]
    after: list[OrientedBox]


def _link_ends(cspace: ConfigurationSpace, validator: _Validator, start: OrientedBox,
               goal: OrientedBox) -> _Ends | str:
    """Links the start and goal poses to the lattice.

    Args:
        cspace: The configuration space.
        validator: Exact checker.
        start: Start pose.
        goal: Goal pose.

    Returns:
        The linked ends, or why one of them cannot be linked.
    """
    start_link = _connect(cspace, validator, start, to_lattice=True)
    goal_link = _connect(cspace, validator, goal, to_lattice=False)
    if start_link is None or goal_link is None:
        side = "start" if start_link is None else "goal"
        return f"The {side} pose is wedged in: no room to move from it."
    (start_state, start_via), (goal_state, goal_via) = start_link, goal_link
    # After an escape, the search runs from the escape pose.
    return _Ends(([start, *start_via][-1], start_state), ([goal, *goal_via][-1], goal_state),
                 [start] if start_via else [], [goal] if goal_via else [])


def _search_with_repairs(scene: Scene, object_id: int, goal: OrientedBox,
                         validator: _Validator, options: MotionOptions) -> MotionPlan:
    """Plans on the lattice, then validates, repairs and smooths the path.

    Args:
        scene: The scene.
        object_id: The object to move.
        goal: Its target box; known to be feasible.
        validator: Exact checker.
        options: Query parameters.

    Returns:
        The plan.
    """
    start = scene[object_id].box
    cspace = ConfigurationSpace.build(scene, object_id, options.yaw_bins, options.clearance,
                                      options.floor_margin, comfort=options.comfort)
    ends = _link_ends(cspace, validator, start, goal)
    if isinstance(ends, str):
        return MotionPlan(MotionStatus.NO_PATH, object_id, message=ends)

    roomy = _Validator.create(scene, object_id, options.clearance + options.comfort,
                              options.floor_margin)
    # Turning one bin moves the footprint corners by an arc of this length.
    search = _Search(cspace, 0.5 * float(np.hypot(*start.size[:2])) * math.pi / options.yaw_bins,
                     options.max_expansions)
    best, exhausted = _Repairer(cspace, search, validator, roomy).run(
        ends.start, ends.goal, options.max_repairs)
    if best is not None:
        best = [*ends.before, *best, *ends.after]
        return _success(scene, object_id, _shortcut(_simplify(best), roomy), search.expansions)
    if exhausted:
        return MotionPlan(MotionStatus.NO_PATH, object_id, expansions=search.expansions,
                          message="No collision-free motion exists at this resolution.")
    return MotionPlan(MotionStatus.SEARCH_LIMIT_REACHED, object_id, expansions=search.expansions,
                      message="The search budget ran out before reaching a conclusion.")


def plan_motion(scene: Scene, object_id: int, goal: OrientedBox,
                options: MotionOptions | None = None) -> MotionPlan:
    """Searches for a collision-free motion of an object to a goal pose.

    Args:
        scene: Current scene; must have a floor map.
        object_id: The object to move.
        goal: Target box of the object (same size as its current box).
        options: Query parameters.

    Returns:
        The plan. ``NO_PATH`` means no path exists at this lattice resolution;
        ``SEARCH_LIMIT_REACHED`` means the search could not conclude.
    """
    options = options or MotionOptions()
    validator = _Validator.create(scene, object_id, options.clearance, options.floor_margin)
    if not validator.pose_ok(scene[object_id].box):
        return MotionPlan(MotionStatus.INVALID_START, object_id,
                          message="The object's current pose already has conflicts.")
    if not validator.pose_ok(goal):
        report = check_placement(scene, object_id, goal, options.clearance, options.floor_margin)
        reasons = "; ".join(c.describe() for c in report.conflicts)
        return MotionPlan(MotionStatus.INVALID_GOAL, object_id,
                          message=f"The goal pose is infeasible: it {reasons}.")
    plan = _search_with_repairs(scene, object_id, goal, validator, options)
    if plan.status is MotionStatus.NO_PATH and options.refine_yaw_bins > options.yaw_bins:
        finer = dataclasses.replace(options, yaw_bins=options.refine_yaw_bins)
        refined = _search_with_repairs(scene, object_id, goal, validator, finer)
        expansions = plan.expansions + refined.expansions
        message = refined.message
        if refined.status is MotionStatus.NO_PATH:
            degrees = 180.0 / options.refine_yaw_bins
            message = f"No collision-free motion exists, even with {degrees:.1f}-degree turns."
        plan = dataclasses.replace(refined, expansions=expansions, message=message)
    return plan



def straight_motion(scene: Scene, object_id: int, goal: OrientedBox,
                    options: MotionOptions | None = None) -> MotionPlan | None:
    """Checks a straight slide (with a steady turn) to a goal, without searching.

    Args:
        scene: The scene.
        object_id: The object.
        goal: Its target box.
        options: Motion parameters (clearance, floor margin).

    Returns:
        The one-segment plan, or None if the slide is not collision-free.
    """
    options = options or MotionOptions()
    validator = _Validator.create(scene, object_id, options.clearance, options.floor_margin)
    if not validator.pose_ok(goal) or not validator.segment_ok(scene[object_id].box, goal):
        return None
    return _success(scene, object_id, [scene[object_id].box, goal], expansions=0)

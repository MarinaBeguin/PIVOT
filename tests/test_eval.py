"""Tests of the evaluation: task predicates, judging and the four variants."""

from pathlib import Path

import pytest

from pivot.eval.run import read_results, run, summarize, wilson_interval
from pivot.eval.tasks import Task, TaskKind, generate_tasks, load_tasks, save_tasks, solve
from pivot.eval.variants import (
    Outcome, judge, llm_check, llm_only, parse_pose_answer, pivot, planner_only, replay,
    replay_failure,
)
from pivot.planning.motion import MotionPlan, MotionStatus
from pivot.meta.client import MetaConfig
from pivot.scene.examples import box, legged_table, office_scene, rectangle_floor
from pivot.scene.graph import Scene, SceneObject
from test_agent import ScriptedModel, message, tool_call

CONFIG = MetaConfig(api_key="test")
# The chair (#3) next to the shelf (#4), whose front spans x in [3.5, 4.5] at y = 3.6.
NEXT_TO_SHELF = Task("a", TaskKind.NEXT_TO, "Move the office chair (#3) next to the shelf (#4).",
                     3, anchor_id=4)
BESIDE_SHELF = (4.0, 3.2, 0.0)
IN_DESK = (1.5, 2.5, 0.0)


def test_a_reachable_pose_next_to_the_anchor_succeeds() -> None:
    verdict = judge(office_scene(), NEXT_TO_SHELF, Outcome(BESIDE_SHELF))
    assert verdict.success and verdict.reachable and verdict.satisfies


def test_a_pose_inside_furniture_is_a_false_claim() -> None:
    verdict = judge(office_scene(), NEXT_TO_SHELF, Outcome(IN_DESK))
    assert not verdict.success and not verdict.reachable and verdict.false_claim


def test_keeping_openings_clear_rejects_a_pose_in_front_of_the_window() -> None:
    clear = Task("b", TaskKind.NEXT_TO_CLEAR, "...", 3, anchor_id=4)
    # Against the shelf's left side, but in front of the window (x in [2.2, 3.4]).
    in_front = (3.15, 3.6, 0.0)
    assert judge(office_scene(), NEXT_TO_SHELF, Outcome(in_front)).success
    verdict = judge(office_scene(), clear, Outcome(in_front))
    assert verdict.reachable and not verdict.satisfies
    assert judge(office_scene(), clear, Outcome(BESIDE_SHELF)).success


def test_a_refusal_is_right_only_for_infeasible_tasks() -> None:
    impossible = Task("c", TaskKind.AT_POINT, "...", 3, point=(1.5, 2.5), feasible=False)
    assert judge(office_scene(), impossible, Outcome(None)).success
    assert not judge(office_scene(), NEXT_TO_SHELF, Outcome(None)).success


def test_planner_only_finds_a_spot_next_to_the_anchor() -> None:
    outcome = planner_only(office_scene(), NEXT_TO_SHELF)
    assert outcome.pose is not None and 1 < outcome.checks <= 8
    assert judge(office_scene(), NEXT_TO_SHELF, outcome).success


def test_parse_pose_answer_reads_json_inside_text() -> None:
    answer = parse_pose_answer('Sure: {"action": "move", "x": 1, "y": 2, "yaw_deg": 90}')
    assert (answer["x"], answer["y"], answer["yaw_deg"]) == (1, 2, 90)
    assert parse_pose_answer("I cannot do that.")["action"] == "refuse"
    assert parse_pose_answer('{"action": "move", "x": 1}')["action"] == "refuse"


def test_llm_only_answers_without_checks() -> None:
    model = ScriptedModel([message('{"action": "move", "x": 1.5, "y": 2.5}')])
    outcome = llm_only(office_scene(), NEXT_TO_SHELF, CONFIG, model)  # type: ignore[arg-type]
    assert outcome.pose == (1.5, 2.5, 0.0) and outcome.checks == 0
    # The model was given the object list.
    assert '"shelf"' in model.requests[0]["messages"][0]["content"]


def test_llm_check_only_reports_pass_or_fail() -> None:
    model = ScriptedModel([message('{"action": "move", "x": 1.5, "y": 2.5}'),
                           message('{"action": "move", "x": 4.0, "y": 3.2, "yaw_deg": 0}')])
    outcome = llm_check(office_scene(), NEXT_TO_SHELF, CONFIG, model)  # type: ignore[arg-type]
    assert outcome.pose == BESIDE_SHELF and outcome.checks == 2
    # The feedback holds no measurement (the list is shared, so it was followed by the reply).
    assert model.requests[1]["messages"][-2]["content"] == (
        "That pose is infeasible. Propose another one, or refuse.")


def test_pivot_reports_where_the_agent_moved_the_object() -> None:
    model = ScriptedModel([
        message(calls=[tool_call("a", "move_object", object_id=3, x=4.0, y=3.2, yaw_deg=0)]),
        message("Done."),
    ])
    outcome = pivot(office_scene(), NEXT_TO_SHELF, CONFIG, model)  # type: ignore[arg-type]
    assert outcome.pose == pytest.approx(BESIDE_SHELF)
    assert outcome.checks == 1 and outcome.note == "Done."
    assert judge(office_scene(), NEXT_TO_SHELF, outcome).success


def test_replay_rejects_a_motion_through_furniture() -> None:
    scene = office_scene()
    start = scene[3].box
    # Straight through the desk (#1), from below it to above it.
    through = MotionPlan(MotionStatus.SUCCESS, 3, waypoints=[start, start.moved(
        start.center[:2] + [0.0, 1.6])])
    assert replay(scene, [through]) is None
    assert replay_failure(scene, [through]).startswith("motion 0 (#3), segment 0")
    verdict = judge(scene, NEXT_TO_SHELF, Outcome((1.5, 3.3, 0.0), motions=[through]))
    assert verdict.false_claim


def test_generated_tasks_are_labeled_and_round_trip(tmp_path: Path) -> None:
    tasks = generate_tasks(office_scene(), count=6, seed=1)
    # Nothing light enough to put on the desk: that kind is skipped.
    assert [t.kind for t in tasks] == [k for k in TaskKind if k is not TaskKind.ON_SURFACE]
    assert all(t.object_id in (1, 3, 4) for t in tasks)  # Not the lamp nor the window.
    save_tasks(tasks, tmp_path / "tasks.json")
    assert load_tasks(tmp_path / "tasks.json") == tasks


def test_a_chair_pushed_under_a_desk_is_judged_by_its_overlap() -> None:
    # A low chair in front of a legged table, in a 4 x 4 m room.
    table = legged_table(1, (2.0, 2.5), size=(1.4, 0.8, 0.76))
    chair = SceneObject(2, "chair", box((2.0, 1.4, 0.3), (0.45, 0.45, 0.6)))
    scene = Scene.from_objects([table, chair], rectangle_floor(4.0, 4.0))
    task = Task("d", TaskKind.TUCK_UNDER, "...", 2, anchor_id=1)
    solved = solve(scene, task)
    assert solved is not None
    center = solved.scene[2].box.center
    outcome = Outcome((center[0], center[1], 0.0), motions=list(solved.motions))
    assert judge(scene, task, outcome).success
    # In front of the table is not under it.
    assert not judge(scene, task, Outcome((2.0, 1.6, 0.0))).success


def test_putting_on_a_desk_is_judged_with_a_carried_move() -> None:
    task = Task("e", TaskKind.ON_SURFACE, "...", 6, anchor_id=1)
    carton = SceneObject(6, "carton", box((3.5, 1.0, 0.15), (0.3, 0.3, 0.3)))
    scene = Scene.from_objects([*office_scene().objects.values(), carton], office_scene().floor)
    # A pose given in words, over the desk: reached by carrying the carton there.
    assert judge(scene, task, Outcome((1.8, 2.5, 0.0))).success
    assert not judge(scene, task, Outcome((3.0, 1.0, 0.0))).success


def test_run_resumes_and_summarizes(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"
    run(office_scene(), [NEXT_TO_SHELF], ["planner_only"], out)
    run(office_scene(), [NEXT_TO_SHELF], ["planner_only"], out)  # Already done: skipped.
    records = read_results(out)
    assert len(records) == 1 and records[0]["success"]
    assert "| planner_only | 1/1 | 1/1 | - | 0/1 |" in summarize(records)


def test_wilson_interval_brackets_the_rate() -> None:
    low, high = wilson_interval(30, 30)
    assert high == 1.0 and 0.85 < low < 0.92
    low, high = wilson_interval(15, 30)
    assert low < 0.5 < high


def test_a_false_claim_into_furniture_reports_its_depth() -> None:
    verdict = judge(office_scene(), NEXT_TO_SHELF, Outcome(IN_DESK))
    assert verdict.collision_m > 0.3  # The chair's center inside the desk.

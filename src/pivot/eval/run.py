"""Runs the evaluation: tasks × variants, judged with exact geometry.

Generate the tasks once (geometry only, no API), then run the variants; the
ones using a model need API access::

    python -m pivot.eval generate --scene scene.json --tasks tasks.json --count 30
    python -m pivot.eval run --scene scene.json --tasks tasks.json --out results.jsonl \\
        --variants planner_only llm_only llm_check pivot

Results are appended one line per (task, variant), so an interrupted run
resumes where it stopped. ``summary`` prints the comparison table.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from pivot.eval.tasks import Task, generate_tasks, load_tasks, save_tasks
from pivot.eval.variants import (
    Outcome, judge, llm_check, llm_only, pivot, planner_only, timed,
)
from pivot.meta.client import MetaConfig
from pivot.scene.graph import Scene

VARIANTS = ("llm_only", "planner_only", "llm_check", "pivot")


def _runner(name: str, scene: Scene, task: Task, budget: int,
            config: MetaConfig | None) -> Callable[[], Outcome]:
    """Binds a variant to its arguments.

    Args:
        name: Variant name, see :data:`VARIANTS`.
        scene: The initial scene.
        task: The task.
        budget: Geometric check budget.
        config: API configuration; only needed by the model variants.

    Returns:
        The variant, ready to run.
    """
    if name == "planner_only":
        return lambda: planner_only(scene, task, budget)
    assert config is not None, f"{name} needs API access"
    if name == "llm_only":
        return lambda: llm_only(scene, task, config)
    if name == "llm_check":
        return lambda: llm_check(scene, task, config, budget=budget)
    return lambda: pivot(scene, task, config, budget=budget)


def run(scene: Scene, tasks: list[Task], variants: list[str], out: Path, budget: int = 8,
        config: MetaConfig | None = None, tag: str = "") -> None:
    """Runs and judges every (task, variant) pair not already in the output.

    Args:
        scene: The initial scene.
        tasks: The tasks.
        variants: Variant names.
        out: JSON-lines result file, appended to.
        budget: Geometric check budget of the checked variants.
        config: API configuration; only needed by the model variants.
        tag: Appended to the variant names in the results, to compare models
            (``pivot`` with Muse, ``pivot_llama`` with Llama).
    """
    done = {(r["task_id"], r["variant"]) for r in read_results(out)}
    with out.open("a", encoding="utf-8") as file:
        for task in tasks:
            for name in variants:
                if (task.task_id, name + tag) in done:
                    continue
                try:
                    outcome = timed(_runner(name, scene, task, budget, config))
                except Exception as error:  # pylint: disable=broad-exception-caught
                    # One failed API call must not stop a long run; it counts as a failure.
                    outcome = Outcome(None, note=f"error: {error}")
                verdict = judge(scene, task, outcome)
                record = {"task_id": task.task_id, "variant": name + tag,
                          "kind": task.kind.value,
                          "feasible": task.feasible, "pose": outcome.pose,
                          "checks": outcome.checks, "seconds": round(outcome.seconds, 2),
                          "success": verdict.success, "claimed": verdict.claimed,
                          "reachable": verdict.reachable, "satisfies": verdict.satisfies,
                          "collision_m": round(verdict.collision_m, 3),
                          "judge_note": verdict.detail,
                          "model_calls": outcome.usage.calls,
                          "prompt_tokens": outcome.usage.prompt_tokens,
                          "completion_tokens": outcome.usage.completion_tokens,
                          "motions": [m.to_dict() for m in outcome.motions],
                          "note": outcome.note[:500]}
                file.write(json.dumps(record) + "\n")
                file.flush()
                print(f"{task.task_id} {name + tag:13s} {'ok ' if verdict.success else 'FAIL'} "
                      f"checks={outcome.checks} {outcome.seconds:.1f}s", flush=True)


def read_results(path: Path) -> list[dict[str, Any]]:
    """Reads a result file.

    Args:
        path: JSON-lines file; may not exist yet.

    Returns:
        The records.
    """
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _variants(records: list[dict[str, Any]]) -> list[str]:
    """Lists the variants present in results: the base ones first, then tagged ones.

    Args:
        records: Result records.

    Returns:
        Variant names.
    """
    present = list(dict.fromkeys(r["variant"] for r in records))
    return [n for n in VARIANTS if n in present] + [n for n in present if n not in VARIANTS]


def _rate(rows: list[dict[str, Any]], key: str) -> str:
    """Formats how many records have a true field.

    Args:
        rows: Result records.
        key: Boolean field.

    Returns:
        E.g. ``"12/15"``, or ``"-"`` without records.
    """
    return f"{sum(r[key] for r in rows)}/{len(rows)}" if rows else "-"


def summarize(records: list[dict[str, Any]]) -> str:
    """Builds the comparison table.

    Args:
        records: Result records.

    Returns:
        Markdown tables: success overall and on feasible / infeasible tasks,
        false claims, mean checks and time per variant; then success per
        task kind.
    """
    lines = ["| Variant | Success | Feasible tasks | Infeasible tasks | False claims "
             "| Checks | Time (s) |", "|---|---|---|---|---|---|---|"]

    for name in _variants(records):
        rows = [r for r in records if r["variant"] == name]
        if not rows:
            continue
        false_claims = [{"x": r["claimed"] and not r["success"]} for r in rows]
        lines.append(
            f"| {name} | {_rate(rows, 'success')} "
            f"| {_rate([r for r in rows if r['feasible']], 'success')} "
            f"| {_rate([r for r in rows if not r['feasible']], 'success')} "
            f"| {_rate(false_claims, 'x')} "
            f"| {sum(r['checks'] for r in rows) / len(rows):.1f} "
            f"| {sum(r['seconds'] for r in rows) / len(rows):.1f} |")
    return "\n".join(lines) + "\n\n" + _by_kind(records) + "\n\n" + _cost(records)


# A false claim (the user moves furniture into a collision, or blocks a door)
# is taken as this many times costlier than a false refusal (a missed move).
FALSE_CLAIM_COST = 10.0


def _cost(records: list[dict[str, Any]]) -> str:
    """Builds the table of the cost of being wrong, and of being slow.

    Args:
        records: Result records.

    Returns:
        A Markdown table per variant: success rate with its 95% Wilson
        interval, false claims (and how deep the claimed poses collide),
        false refusals, expected cost per request, latency, throughput and
        model usage.
    """
    lines = ["| Variant | Success (95% CI) | False claims | Collision depth (cm) "
             "| False refusals | Expected cost | Latency p50 / p95 (s) | Tasks per hour "
             "| Model calls | Tokens in / out |", "|---" * 10 + "|"]
    for name in _variants(records):
        rows = [r for r in records if r["variant"] == name]
        if not rows:
            continue
        n = len(rows)
        low, high = wilson_interval(sum(r["success"] for r in rows), n)
        claims = [r for r in rows if r["claimed"] and not r["success"]]
        refusals = sum(r["feasible"] and not r["claimed"] for r in rows)
        depths = [100 * r.get("collision_m", 0.0) for r in claims if r.get("collision_m", 0.0)]
        seconds = np.array([r["seconds"] for r in rows])
        cost = (FALSE_CLAIM_COST * len(claims) + refusals) / n
        depth = f"{np.mean(depths):.0f} (n={len(depths)})" if depths else "-"
        lines.append(
            f"| {name} | {100 * sum(r['success'] for r in rows) / n:.0f}% "
            f"({100 * low:.0f}–{100 * high:.0f}%) | {len(claims)}/{n} | {depth} "
            f"| {refusals}/{n} | {cost:.2f} "
            f"| {np.median(seconds):.1f} / {np.percentile(seconds, 95):.1f} "
            f"| {3600 / max(seconds.mean(), 1e-9):.0f} "
            f"| {np.mean([r.get('model_calls', 0) for r in rows]):.1f} "
            f"| {np.mean([r.get('prompt_tokens', 0) for r in rows]):.0f} / "
            f"{np.mean([r.get('completion_tokens', 0) for r in rows]):.0f} |")
    return "\n".join(lines)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Computes the Wilson score interval of a success rate.

    Args:
        successes: Number of successes.
        n: Number of trials; positive.
        z: Normal quantile (1.96 for 95%).

    Returns:
        The interval's bounds, in [0, 1].
    """
    rate = successes / n
    center = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, center - half), min(1.0, center + half)


def _by_kind(records: list[dict[str, Any]]) -> str:
    """Builds the table of successes per task kind and variant.

    Args:
        records: Result records.

    Returns:
        A Markdown table.
    """
    names = _variants(records)
    lines = ["| Kind | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for kind in dict.fromkeys(r["kind"] for r in records):
        cells = [_rate([r for r in records if r["kind"] == kind and r["variant"] == n],
                       "success") for n in names]
        lines.append(f"| {kind} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Args:
        argv: Command-line arguments; ``sys.argv`` if None.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="make and label the tasks")
    generate.add_argument("--scene", type=Path, required=True)
    generate.add_argument("--tasks", type=Path, required=True)
    generate.add_argument("--count", type=int, default=36)
    generate.add_argument("--seed", type=int, default=0)
    evaluate = commands.add_parser("run", help="run and judge the variants")
    evaluate.add_argument("--scene", type=Path, required=True)
    evaluate.add_argument("--tasks", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    evaluate.add_argument("--budget", type=int, default=8)
    evaluate.add_argument("--env-file", type=Path, default=Path(".env"))
    evaluate.add_argument("--tag", default="", help="suffix of the variant names, e.g. _llama")
    summary = commands.add_parser("summary", help="print the comparison table")
    summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "generate":
        tasks = generate_tasks(Scene.load(args.scene), args.count, args.seed)
        save_tasks(tasks, args.tasks)
        print(f"{len(tasks)} tasks, {sum(t.feasible for t in tasks)} feasible -> {args.tasks}")
    elif args.command == "run":
        needs_api = any(v != "planner_only" for v in args.variants)
        config = MetaConfig.from_env(args.env_file) if needs_api else None
        run(Scene.load(args.scene), load_tasks(args.tasks), args.variants, args.out,
            args.budget, config, args.tag)
        print(summarize(read_results(args.out)))
    else:
        print(summarize(read_results(args.out)))

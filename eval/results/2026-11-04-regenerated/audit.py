"""Recompute counts from the preserved evaluation records; no models are called."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
ROOMS = ("office", "bedroom", "classroom", "apartment")
VARIANTS = ("llm_only", "llm_check", "pivot", "planner_only")


def wilson(wins, total):
    rate = wins / total
    z = 1.96
    centre = (rate + z*z/(2*total)) / (1 + z*z/total)
    radius = z * math.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / (1+z*z/total)
    return [round(centre-radius, 6), round(centre+radius, 6)]


def counts(rows):
    answer = {}
    for variant in VARIANTS:
        subset = [r for r in rows if r["variant"] == variant]
        errors = [r for r in subset if r.get("note", "").startswith("error:")]
        wins = sum(r["success"] and not r.get("note", "").startswith("error:") for r in subset)
        answer[variant] = {
            "tasks": len(subset), "successes": wins,
            "feasible_tasks": sum(r["feasible"] for r in subset),
            "feasible_successes": sum(r["feasible"] and r["success"] and not r.get("note", "").startswith("error:") for r in subset),
            "infeasible_tasks": sum(not r["feasible"] for r in subset),
            "infeasible_successes": sum(not r["feasible"] and r["success"] and not r.get("note", "").startswith("error:") for r in subset),
            "success_rate": round(wins/len(subset), 6),
            "success_ci95": wilson(wins, len(subset)),
            "false_claims": sum(r["claimed"] and not r["success"] for r in subset),
            "false_refusals": sum(r["feasible"] and not r["claimed"] for r in subset),
            "explicit_errors": len(errors),
            "original_successes": sum(r["success"] for r in subset),
        }
    return answer


def main():
    all_rows = []
    report = {"rooms": {}}
    for room in ROOMS:
        folder = ROOT / room
        tasks = json.loads((folder / "tasks.json").read_text())
        rows = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
        task_by_id = {t["task_id"]: t for t in tasks}
        assert len(task_by_id) == len(tasks)
        pairs = Counter((r["task_id"], r["variant"]) for r in rows)
        assert set(pairs) == {(t, v) for t in task_by_id for v in VARIANTS}
        assert set(pairs.values()) == {1}
        summaries = json.loads((folder / "summary.json").read_text())["variants"]
        for row in rows:
            task = task_by_id[row["task_id"]]
            assert (row["kind"], row["feasible"]) == (task["kind"], task["feasible"])
        measured = counts(rows)
        for variant, c in measured.items():
            saved = summaries[variant]
            assert saved["tasks"] == c["tasks"]
            assert saved["success"] == round(c["original_successes"]/c["tasks"], 3)
            assert saved["false_claims"] == c["false_claims"]
            assert saved["false_refusals"] == c["false_refusals"]
        report["rooms"][room] = {"tasks": len(tasks), "records": len(rows), "variants": measured}
        all_rows.extend(dict(r, room=room) for r in rows)
    report["all_rooms"] = counts(all_rows)
    report["other_three_rooms"] = counts([r for r in all_rows if r["room"] != "office"])
    cases = list(ET.parse(ROOT / "safety/safety.xml").getroot().iter("testcase"))
    failures = sum(c.find("failure") is not None or c.find("error") is not None for c in cases)
    skipped = sum(c.find("skipped") is not None for c in cases)
    report["safety"] = {"cases": len(cases), "passed": len(cases)-failures-skipped,
                        "failed": failures, "skipped_or_expected_failure": skipped}
    assert report["safety"] == {"cases": 42, "passed": 41, "failed": 0, "skipped_or_expected_failure": 1}
    (ROOT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    sums = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS" and "__pycache__" not in path.parts:
            sums.append(hashlib.sha256(path.read_bytes()).hexdigest()+"  "+str(path.relative_to(ROOT)))
    (ROOT / "SHA256SUMS").write_text("\n".join(sums)+"\n")
    print("Verified 139 tasks, 556 unique task/variant records and all original summary counts.")
    print(json.dumps(report["all_rooms"], indent=2))


if __name__ == "__main__":
    main()

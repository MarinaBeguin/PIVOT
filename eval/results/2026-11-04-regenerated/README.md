# Regenerated-scene evaluation archive

Run id: `eval_20261104_155715`. Documentation added November 5, 2026.
This date describes the documentation update, not a backdated commit or PR.

Each room contains unchanged `tasks.json`, `results.jsonl`, `summary.json`, `summary.md`
and `scene.sha256` copied from the recorded run. Scene snapshots were checked against
those fingerprints before publication; clouds and scene files are not duplicated here.
`provenance.json` states what is and is not recorded. No launch code commit or exact
model version was captured, so a frozen-code held-out claim is not established.

The separate `audit.json` counts explicit errors as failed requests. Original summaries
instead count one classroom Muse-alone timeout as a correct refusal on an infeasible task.
This changes Muse alone from 48/139 to 47/139 and leaves all other variants unchanged.
The record is `classroom/t021`, variant `llm_only`; no record is removed or rewritten.

Run `python eval/results/2026-11-04-regenerated/audit.py` from the repository root to
check all 556 task/variant records and regenerate the adjusted counts and checksums.
Verify checksums with `sha256sum -c SHA256SUMS` from this directory.
The script checks count agreement with saved summaries; it does not rerun models,
rejudge trajectories, verify missing scene files or establish physical correctness.

The safety folder preserves the separate CPU-only report (41 passed, 1 expected failure).
Its skipped/xfail case is undo of a confirmed route when no furniture was moved.
Synthetic/scripted tests do not validate live language understanding or browser rendering.

See [main results and limitations](../../../README.md#results) and
[evaluation protocol](../../README.md).

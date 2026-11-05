# Safety evaluation

Run from the code checkout to evaluate:

```bash
bash ~/code/PIVOT/eval/run_safety.sh "$PIVOT_EVAL_OUT/safety"
```

Uses the current checkout's src and tests; PIVOT_EVAL_CODE_ROOT overrides it.
PIVOT_PYTHON overrides the cluster interpreter. No model API calls or scene writes.
Reports: safety.log and JUnit safety.xml. Nonzero exit means an unexpected failure.

Coverage: final access routes and doorway safety, rejected edits, provisional route
rollback, alternative selection followed by confirmation and undo, printable route
presence, step-budget rollback, and motion search exhaustion distinct from no-path.
Fixtures are synthetic, and model replies are scripted. These are contract
regressions, not evidence of live model understanding or browser rendering.
The regenerated-scene benchmark and manual viewer review remain separate.

Known limitation explicitly tracked as strict xfail: confirming a route without
moving furniture creates no undo entry, so undo does not remove that route.
An unexpected pass requires updating this recorded limitation.

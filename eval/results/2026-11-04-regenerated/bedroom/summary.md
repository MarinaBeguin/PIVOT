| Variant | Success | Feasible tasks | Infeasible tasks | False claims | Checks | Time (s) |
|---|---|---|---|---|---|---|
| llm_only | 7/25 | 3/8 | 4/17 | 18/25 | 0.0 | 40.4 |
| planner_only | 25/25 | 8/8 | 17/17 | 0/25 | 1.0 | 0.7 |
| llm_check | 18/25 | 5/8 | 13/17 | 5/25 | 3.0 | 320.0 |
| pivot | 25/25 | 8/8 | 17/17 | 0/25 | 8.0 | 45.6 |

| Kind | llm_only | planner_only | llm_check | pivot |
|---|---|---|---|---|
| at_point | 3/4 | 4/4 | 4/4 | 4/4 |
| tuck_under | 0/3 | 3/3 | 3/3 | 3/3 |
| on_surface | 0/4 | 4/4 | 2/4 | 4/4 |
| at_point_blocked | 2/4 | 4/4 | 3/4 | 4/4 |
| turn_in_place | 1/4 | 4/4 | 3/4 | 4/4 |
| clear_door | 0/1 | 1/1 | 1/1 | 1/1 |
| against_wall | 0/4 | 4/4 | 1/4 | 4/4 |
| wheelchair_way | 1/1 | 1/1 | 1/1 | 1/1 |

| Variant | Success (95% CI) | False claims | Collision depth (cm) | False refusals | Expected cost | Latency p50 / p95 (s) | Tasks per hour | Model calls | Tokens in / out |
|---|---|---|---|---|---|---|---|---|---|
| llm_only | 28% (14–48%) | 18/25 | 24 (n=11) | 0/25 | 7.20 | 28.9 / 74.5 | 89 | 1.0 | 2230 / 5037 |
| planner_only | 100% (87–100%) | 0/25 | - | 0/25 | 0.00 | 0.2 / 2.7 | 5344 | 0.0 | 0 / 0 |
| llm_check | 72% (52–86%) | 5/25 | - | 2/25 | 2.08 | 170.4 / 815.2 | 11 | 3.6 | 8411 / 42626 |
| pivot | 100% (87–100%) | 0/25 | - | 0/25 | 0.00 | 26.6 / 118.1 | 79 | 8.9 | 77773 / 4153 |

| Variant | Success | Feasible tasks | Infeasible tasks | False claims | Checks | Time (s) |
|---|---|---|---|---|---|---|
| llm_only | 13/40 | 6/11 | 7/29 | 27/40 | 0.0 | 66.5 |
| planner_only | 40/40 | 11/11 | 29/29 | 0/40 | 1.6 | 17.3 |
| llm_check | 36/40 | 7/11 | 29/29 | 2/40 | 3.3 | 393.9 |
| pivot | 39/40 | 10/11 | 29/29 | 0/40 | 8.2 | 66.5 |

| Kind | llm_only | planner_only | llm_check | pivot |
|---|---|---|---|---|
| next_to | 1/4 | 4/4 | 4/4 | 4/4 |
| next_to_clear | 1/4 | 4/4 | 4/4 | 4/4 |
| at_point | 3/4 | 4/4 | 4/4 | 4/4 |
| tuck_under | 1/4 | 4/4 | 4/4 | 4/4 |
| on_surface | 1/4 | 4/4 | 4/4 | 4/4 |
| at_point_blocked | 1/4 | 4/4 | 4/4 | 4/4 |
| turn_in_place | 0/4 | 4/4 | 4/4 | 4/4 |
| clear_door | 3/4 | 4/4 | 3/4 | 4/4 |
| against_wall | 0/4 | 4/4 | 2/4 | 3/4 |
| wheelchair_way | 2/4 | 4/4 | 3/4 | 4/4 |

| Variant | Success (95% CI) | False claims | Collision depth (cm) | False refusals | Expected cost | Latency p50 / p95 (s) | Tasks per hour | Model calls | Tokens in / out |
|---|---|---|---|---|---|---|---|---|---|
| llm_only | 32% (20–48%) | 27/40 | 23 (n=10) | 0/40 | 6.75 | 37.5 / 232.8 | 54 | 1.0 | 7720 / 8502 |
| planner_only | 100% (91–100%) | 0/40 | - | 0/40 | 0.00 | 4.0 / 123.4 | 208 | 0.0 | 0 / 0 |
| llm_check | 90% (77–96%) | 2/40 | - | 2/40 | 0.55 | 349.0 / 903.9 | 9 | 3.9 | 30982 / 50971 |
| pivot | 98% (87–100%) | 0/40 | - | 1/40 | 0.03 | 46.6 / 163.5 | 54 | 8.7 | 123063 / 4312 |

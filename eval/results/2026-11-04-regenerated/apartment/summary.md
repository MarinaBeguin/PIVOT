| Variant | Success | Feasible tasks | Infeasible tasks | False claims | Checks | Time (s) |
|---|---|---|---|---|---|---|
| llm_only | 10/35 | 3/13 | 7/22 | 25/35 | 0.0 | 37.3 |
| planner_only | 35/35 | 13/13 | 22/22 | 0/35 | 1.5 | 2.5 |
| llm_check | 31/35 | 11/13 | 20/22 | 3/35 | 2.8 | 267.0 |
| pivot | 32/35 | 10/13 | 22/22 | 2/35 | 4.8 | 34.8 |

| Kind | llm_only | planner_only | llm_check | pivot |
|---|---|---|---|---|
| next_to | 0/4 | 4/4 | 4/4 | 4/4 |
| next_to_clear | 1/4 | 4/4 | 4/4 | 4/4 |
| at_point | 4/4 | 4/4 | 4/4 | 4/4 |
| tuck_under | 1/4 | 4/4 | 4/4 | 3/4 |
| on_surface | 1/4 | 4/4 | 4/4 | 4/4 |
| at_point_blocked | 1/4 | 4/4 | 4/4 | 4/4 |
| turn_in_place | 0/4 | 4/4 | 3/4 | 4/4 |
| against_wall | 0/4 | 4/4 | 2/4 | 2/4 |
| wheelchair_way | 2/3 | 3/3 | 2/3 | 3/3 |

| Variant | Success (95% CI) | False claims | Collision depth (cm) | False refusals | Expected cost | Latency p50 / p95 (s) | Tasks per hour | Model calls | Tokens in / out |
|---|---|---|---|---|---|---|---|---|---|
| llm_only | 29% (16–45%) | 25/35 | 37 (n=10) | 0/35 | 7.14 | 23.9 / 83.1 | 96 | 1.0 | 3419 / 4561 |
| planner_only | 100% (90–100%) | 0/35 | - | 0/35 | 0.00 | 0.3 / 8.2 | 1448 | 0.0 | 0 / 0 |
| llm_check | 89% (74–95%) | 3/35 | - | 1/35 | 0.89 | 150.9 / 854.4 | 13 | 3.3 | 11798 / 31945 |
| pivot | 91% (78–97%) | 2/35 | - | 1/35 | 0.60 | 27.5 / 78.3 | 103 | 6.5 | 58842 / 2702 |

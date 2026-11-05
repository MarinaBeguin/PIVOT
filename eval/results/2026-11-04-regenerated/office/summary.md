| Variant | Success | Feasible tasks | Infeasible tasks | False claims | Checks | Time (s) |
|---|---|---|---|---|---|---|
| llm_only | 18/39 | 13/14 | 5/25 | 21/39 | 0.0 | 46.9 |
| planner_only | 39/39 | 14/14 | 25/25 | 0/39 | 1.3 | 1.8 |
| llm_check | 30/39 | 11/14 | 19/25 | 9/39 | 2.3 | 189.8 |
| pivot | 38/39 | 13/14 | 25/25 | 1/39 | 2.9 | 27.0 |

| Kind | llm_only | planner_only | llm_check | pivot |
|---|---|---|---|---|
| next_to | 4/4 | 4/4 | 4/4 | 4/4 |
| next_to_clear | 4/4 | 4/4 | 4/4 | 4/4 |
| at_point | 4/4 | 4/4 | 4/4 | 4/4 |
| tuck_under | 0/4 | 4/4 | 3/4 | 4/4 |
| on_surface | 0/4 | 4/4 | 3/4 | 4/4 |
| at_point_blocked | 2/4 | 4/4 | 3/4 | 4/4 |
| to_room | 0/4 | 4/4 | 4/4 | 4/4 |
| turn_in_place | 1/4 | 4/4 | 2/4 | 4/4 |
| against_wall | 3/4 | 4/4 | 1/4 | 3/4 |
| wheelchair_way | 0/3 | 3/3 | 2/3 | 3/3 |

| Variant | Success (95% CI) | False claims | Collision depth (cm) | False refusals | Expected cost | Latency p50 / p95 (s) | Tasks per hour | Model calls | Tokens in / out |
|---|---|---|---|---|---|---|---|---|---|
| llm_only | 46% (32–61%) | 21/39 | 21 (n=10) | 0/39 | 5.38 | 29.5 / 204.0 | 77 | 1.0 | 1933 / 5442 |
| planner_only | 100% (91–100%) | 0/39 | - | 0/39 | 0.00 | 0.6 / 8.4 | 2055 | 0.0 | 0 / 0 |
| llm_check | 77% (62–87%) | 9/39 | - | 0/39 | 2.31 | 106.9 / 654.6 | 19 | 2.7 | 5429 / 22737 |
| pivot | 97% (87–100%) | 1/39 | - | 0/39 | 0.26 | 27.0 / 41.4 | 133 | 4.9 | 35592 / 2158 |

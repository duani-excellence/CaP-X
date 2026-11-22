# Graph execution state and budget review, 2026-11-20

Scope: formal CaP-X workspace; no model calls, physical experiments, dependency changes or existing artifact overwrites in this review. The regression cases use public robot/sensor fixtures, not simulator truth. These results establish software behavior, not physical recovery effectiveness.

## Confirmed failures and corrections

1. A close/open/motion callback could change the robot and then raise `ValueError`. The graph recorded `unknown_partial_effects`, but the phase and robot epoch still authorized a new actor to close again, open again, home, or recall an obsolete robot pose. Unknown physical effects now advance the robot epoch, invalidate the unfinished suffix, and set `reconciliation_required` with `unknown_physical_effect_terminal_stop`. Successful RGB-D capture never clears this condition. Earlier returned actions remain immutable historical facts. No controller completion/cancellation receipt exists, so current public encoders alone do not certify the intervening path/contact effects. This case is an explicit unsuccessful terminal exit, not claimed recovery.
2. A returned close followed by failed RGB-D parsing was previously not committed to phase state. A later actor could repeat the close. Returned closure now commits `holding_unverified`, held-object identity and source expiration before camera parsing. It preserves the returned receipt and forbids repeated close/home. It does not certify grasp, lift or task success. Later public evidence can still verify the retained stage.
3. Reusing a shared initial DeepSeek answer gave the second arm more control/recovery time. In a deterministic clock test, a 60-second initial query yielded G0 deadline 200 and G1 deadline 260 under the same 100-second episode cap. The cached arm now charges `initial.generation_seconds` against its policy deadline. Actual wall time still counts the real shared query only once; policy end-to-end time includes the shared latency once per arm.
4. The power CLI silently used six blocks despite the current eight-block runtime configuration. It now requires `--config` or explicit `--max-blocks`, rejects mismatched overrides, stores the configuration hash, and reports resource counts separately from empirical runtime and actual billed cost.

Relevant implementation: `capx/execution/graph_feedback.py`, `capx/execution/grounded_feedback.py`, `capx/experiments/graph_v2.py`, `capx/experiments/graph_power.py`.

## Regression evidence

- Original review: eight reproducible repeated-close/open/home failures. An expanded first run also exposed two budget failures; two motion fixtures initially had an overly strict numeric setup assertion, corrected from greater-than 0.1 m to greater-than 0.09 m before the implementation fix.
- Corrected and expanded state tests: 15 tests, including unknown-motion invalidation, fresh-capture not clearing unknown effects, returned-close camera failure and perception errors not inventing a physical effect.
- State-related regression: **57 passed**. `outputs/reproduction/graph-v2-evaluation-tests-20261120/action-reconciliation-green-v2.xml` and `.log`.
- Budget/runner/legacy graph regression: **83 passed**. `outputs/reproduction/graph-v2-evaluation-tests-20261120/budget-parity-green-v2.xml` and `.log`.
- Red logs are preserved beside the green logs. No failing assertion was removed to make the implementation pass.

Local reproduction:

```bash
cd /home/hanyu/dqb/cap-x-20261106
.venv/bin/python -B -m pytest tests/graph_v2/test_review_partial_effects.py tests/graph_v2/test_review_budgets.py -q -p no:cacheprovider
```

## Resource arithmetic

Normal B0/G0/G1, three tasks, 100 independent scenes/task, two initial draws/scene, eight blocks/group:

| Quantity | Upper count or configured limit |
| --- | ---: |
| Paired units | 600 |
| Episodes | 1,800 |
| Unique initial requests/unit | 2 (B0 own, G0/G1 shared) |
| Replan requests/unit | 21 |
| Unique LLM requests | **13,800** |
| Physical API calls, 192/episode | **345,600** |
| Output tokens, 32,768/request | **452,198,400** |
| Output component at peak $3.96/M | **$1,790.705664** |
| Sum of episode policy caps, 2,400 s/episode | **4,320,000 s = 50 days** |

The old six-block calculation gave 10,200 requests, undercounting by 3,600. The same 20-scene/task development design with two draws needs at most 2,760 requests and 360 episodes. Current example caps of 2,400 requests, $500 and seven days are developer resource stops; they do not cover the formal worst-case envelope.

The output component is neither an invoice nor a full cost upper bound. Input prompt size is not globally capped, and CostLedger requires two-times reservation headroom. Likewise, episode time caps omit environment initialization, source archiving and final video encoding. Empirical mean/p95 extrapolations remain planning estimates, not worst-case guarantees. The power output explicitly marks these unknowns and insufficient caps rather than promising completion.

Planning command after genuine, fixed-scene development records exist (does not call a model):

```bash
.venv/bin/python -B -m capx.experiments.graph_power DEVELOPMENT_RECORDS.jsonl \
  --config env_configs/graph_feedback/development.json \
  --output NEW_POWER_PLAN.json
```

## Existing protections retained

Monitor phase and recovery counters persist across actors. A lift attempt and plan execution are consumed before possibly partial physical effects and are never refunded. Low-level API counters persist across blocks. The sandbox retains its cumulative RPC counter across child restarts. Persistent model request labels and prompt hashes prevent silent re-request of unknown/failed attempts. The shared initial request is billed once globally and represented once logically in each arm.

`max_episode_local` excludes the separately bounded two hold probes and some perception retries; all still consume the physical/API or sandbox limits. Report these counters by their actual definitions, not as a single total recovery counter.

## Freeze additions and remaining limits

All five graph experiment configs now include the table/self-geometry/IK helper source entries, robot-selfseg asset manifest, and its 11 actual URDF/mesh asset paths. Asset bytes were hash-checked before adding paths. Pending helper files must exist before source inventory/freeze can pass. The paired runner's existing source-drift checks remain active.

This review does not establish three-task physical acceptance or authorize formal evaluation from the current development state. Unknown post-dispatch errors currently terminate instead of claiming recoverability. A future no-effect IK rejection needs a trusted before-dispatch receipt and must distinguish failure at a later waypoint after earlier movement. The unchanged independent physical evaluator supplies physical task outcome; the experiment policy additionally requires completion within the frozen execution limits.


## Post-evaluation deadline and policy outcome correction

A read-only deterministic clock reproduction exposed one more accounting defect: a program could return just before its episode deadline, then cross the deadline during the independent stable-state evaluation or public finish verification. The previous runner accepted `done` before checking elapsed time, so a 101-second outcome under a 100-second cap could be reported as success. This is a reproduced software path, not evidence that an existing real episode violated its cap.

The corrected runner distinguishes the unchanged physical outcome from admissible policy completion. `reward`, `info.physical_success`, execution evidence and final physical state remain intact. A completion after the episode or global wall deadline is `status=budget_exceeded`, `policy_success=false`, and `stop_reason=episode_or_global_wall_budget`. Truncation, a terminal monitor exit or a non-OK actor execution cannot become a success merely because an independent physical check is positive. Reaching the allowed last API call and successfully finishing within the limit remains valid; using the budget and exiting because it is exhausted are different conditions.

`policy_elapsed_seconds` includes initial generation (including the shared initial latency charged to each paired arm), control, physical evaluation and public finish verification. Final video encoding and artifact cleanup remain outside the episode policy deadline and inside the separately recorded real elapsed time. Physical evaluation remains unchanged. No simulator-truth fields were added to online feedback.

The statistical success predicate now rejects explicit policy failure, truncation, terminal exits and non-OK raw execution status. A legacy valid model record containing `reward=1,status=success` remains eligible without requiring new fields. Offline episode reports retain `task_success` as the policy outcome and add `policy_success`, `physical_reward`, `physical_task_success`, `physical_success_source`, and `policy_outcome_available`. Handwritten historical fixtures without any status retain their physical result and explicitly lack a policy outcome; they are not silently promoted to completed model episodes. The explicitly physical metric `replan_physical_success_blocks` is unchanged.

Validation was test-first. The focused policy, runner, statistics, offline analysis and budget suite completed **88 passed**, with two existing PyRoKi deprecation warnings. Red and green logs are retained at:

- `outputs/reproduction/graph-v2-evaluation-tests-20261120/policy-outcome-red.xml` and `.log`.
- `outputs/reproduction/graph-v2-evaluation-tests-20261120/policy-outcome-green.xml` and `.log`.

```bash
.venv/bin/python -B -m pytest tests/graph_v2/test_policy_outcome.py \
  tests/graph_v2/test_runner_v2.py tests/graph_v2/test_statistics.py \
  tests/graph_v2/test_offline_analysis.py tests/graph_v2/test_review_budgets.py \
  -q -p no:cacheprovider
```

These tests establish software policy accounting only. They do not establish an improvement in physical success or replace pending real task acceptance and formal paired experiments.

# CaP-X 最终开发 / 正式评测操作清单（只读审查草案）

初稿日期：2026-11-20；2026-11-22按V18有限预算更新。尚未执行下列准备、冻结、Snapshot、付费采集命令；不是 GO 结论。
正式工程：`/home/hanyu/dqb/cap-x-20261106`。当前三任务的最终共同底座仍需完成真实验收。

## 0. 顺序与不变量

顺序：共同底座稳定与三任务真实 smoke → 最终新场景开发 20/task × 2 repeats × 3 groups → 所有结果导出和验收证据 → 功效规划 → 未见正式场景 Snapshot → formal normal 冻结 / 执行 → normal 全量初始结果导出 → 预声明 20/task 子集的三个单机制消融 → 独立压力测试。

最终开发启动后不再修改运行 / 提示词 / 评分 / 分析 / CLI / service 源码。若需要修算法或恢复预算，保留旧开发结果，完成修复后以新目录和新分配重新开始最终开发；不拿旧源结果通过当前源的正式 gate。

当前 per-episode 合同：max_blocks=16, api_budget=192, episode_seconds=2400, max_tokens=32768, request_timeout=400, max_local=6, max_episode_local=24, max_episode_views=8。后续最终修改以最终 config 为准。冻结前核实不变。

目前 development 总预算为 6000 requests / $1000 conservative peak cost / 14 active days；120 units 的请求上限为 5640，覆盖本轮计划。正常 formal 100/task 共 600 units / 1800 episodes，最大 28200 requests；不能沿用开发的 6000 或旧 2400。

## 1. 一次性分配和服务来源

在最终稳定代码下，选一个全新 `$BASE`。以下 shell 变量仅用于分步操作，勿整份无检查直接运行。

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export BASE="$PWD/outputs/model_comparison/graph_final_$(date +%Y%m%d_%H%M%S)"
mkdir "$BASE"
```

先复制完整最终配置为 `$BASE/preparation-config.json`，删除 `snapshot_manifest`、`acceptance_file`、`initial_manifest`。保留 sources/prompt_files/scoring_files/analysis_sources/cli_files/asset_files/robot_calibration_file/services/runtime_environment/allowed_graph_ablations/mechanism_scene_subset。不要把输出目录、生成 manifest、state、frozen.json 添加到 sources。尤其不要让 Snapshot 的 preparation 配置指向自己尚在写的 scenes.json。

当前 service_inventory 为已验 preflight-v4/service-inventory.json，当前 v18-20261122 只作额外运行旁证。正式最终开发可继续引用 v4，只要实际服务 identity 验证通过；不要仅为更新日期替换它。若需要新的 inventory，必须在最终开发之前生成，且后面 formal 使用同一身份。GET/schema/进程检查不调用模型：

```bash
.venv/bin/python -B -m capx.experiments.graph_preflight \
  --config "$BASE/preparation-config.json" --output "$BASE/service-precheck"
```

这个新 inventory 不自动替换配置；先核对 3 服务 HTTP 200、GraspNet seed schema、recursive local closure、文件均早于进程启动、runtime_environment 七项一致。GPU NVML 读不到不能写成 GPU 已验证；保留现有限制。

生成无付费、无仿真副作用的分配计划：

```bash
.venv/bin/python -B - "$BASE" <<'PY'
import json, sys
from pathlib import Path
from capx.experiments.graph_campaign import make_plan
root=Path(sys.argv[1]); config=json.loads((root/'preparation-config.json').read_text())
plan=make_plan(config['tasks'],20,100,2,20261120)
with (root/'allocation.json').open('x') as f: json.dump(plan,f,indent=2)
PY
```

20261120 allocation seed 已只读检查：历史 1203 个 metadata JSON 中提取的 322 个 seed 与上述 60 开发 + 300 正式场景无交集；详见 `/tmp/capx_final_operation_history_audit.json`。这仅是当前候选检查。最终创建之前再核对真实已执行历史。不要把“规划中但未执行”的 formal seeds 当成已经看过任务效果；也不要把同一 plan 的声明误识别为历史污染。历史 0、21 只做调试，不能加入新场景开发或正式推断。

## 2. 固定开发 Snapshot，随后冻结最终开发

```bash
bash scripts/local/graph_experiment.sh prepare-snapshots \
  --config "$BASE/preparation-config.json" --plan "$BASE/allocation.json" \
  --split development --output "$BASE/development-snapshots"
```

必须 complete=true, completed=60, failed=0。Snapshot preparation 可 `--max-scenes K` 在 scene 边界分段，之后同配置同目录 `--resume`；未知 started 绝不自动重跑。failed 保留，不根据后续任务能否成功换种子。

完成后才写 `$BASE/development-config.json`，内容为 preparation-config + 完整绝对路径 snapshot_manifest。然后：

```bash
bash scripts/local/graph_experiment.sh prepare-development \
  --config "$BASE/development-config.json" --output "$BASE/development" \
  --development-scenes 20 --test-scenes 100 --repeats 2 --allocation-seed 20261120
```

无 `--exploratory`。该命令在付费前实际校验 service 身份并冻结。核对 development/plan.json 与 allocation.json 完全一致。之后的命令实际产生费用：

```bash
bash scripts/local/graph_experiment.sh run --campaign "$BASE/development"
# 或先执行固定几个 unit 检查存储、时长；不是按结果过滤，再续完全部：
# ... run --campaign "$BASE/development" --max-units 6
# ... resume --campaign "$BASE/development"
bash scripts/local/graph_experiment.sh status --campaign "$BASE/development"
```

需要正常暂停时在另一终端用 `request-stop --campaign ... --reason ...`；只在完成当前配对 unit 后停止。resume 自动识别已处理的 stop request，不手工删除账本，不对已 done 的失败重新请求。进程突然中断且 unit 或 request 完成情况未知时会 fail closed，必须审计；不要直接删 started 标记。

## 3. 全结果导出与精确 acceptance 格式

开发必须 120 complete units / 360 attempted episodes / 0 uncertain requests / source_integrity_complete=true。各任务 20 个 scene × 两重复都满足实际环境 Snapshot 初始化与真实模型尝试，不要求每个 episode 成功。无环境、0 模型 preflight 系统故障不能充当最终模型开发完成场景。

下面只导出已有产物，不调用 API。acceptance 只挑每个 task 一个成功见证；统计的 records 包含全部失败和全部成功：

```bash
.venv/bin/python -B - "$BASE" <<'PY'
import json, sys
from pathlib import Path
from capx.experiments.graph_campaign import load_campaign,status,records,development_execution_evidence
from capx.experiments.graph_statistics import success
base=Path(sys.argv[1]); root=base/'development'; frozen=load_campaign(root)
progress=status(root); state=json.loads((root/'state.json').read_text())
assert progress['complete_units']==progress['planned_units']==120
assert progress['attempted_episodes']==360 and progress['uncertain_requests']==0
assert progress['source_integrity_complete'] and progress['runtime_accounting_complete']
counts, validated, rejected=development_execution_evidence(root,state,frozen['config'])
assert counts=={task:20 for task in frozen['config']['tasks']}, (counts,rejected)
rows=records(root)
with (base/'development-records.jsonl').open('x') as f:
    for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
witnesses={}
for unit in state['units']:
    for group in ('G0','G1'):
        path=(Path(unit['result']).parent/'episodes'/group/'result.json').resolve()
        row=json.loads(path.read_text())
        if str(path) in validated and row.get('online_vision') is True and row.get('executed') is True and row.get('actual_api_calls',0)>0 and success(row):
            witnesses.setdefault(row['task'],str(path))
assert set(witnesses)==set(frozen['config']['tasks']), witnesses
with (base/'acceptance.json').open('x') as f:
    json.dump({'validated_tasks':frozen['config']['tasks'],'evidence_files':list(witnesses.values())},f,indent=2)
PY
```

不能用 Oracle、自制 summary、B0-only 成功、原reward=1但预算/受控退出，或其他源版本的单独实验作为 acceptance。

## 4. 功效与完整规模预算

```bash
.venv/bin/python -B -m capx.experiments.graph_power \
  "$BASE/development-records.jsonl" --config "$BASE/development/frozen.json" \
  --output "$BASE/power-plan.json" --effect 0.10 --minimum-scenes 100 --repeats 2
```

主效应是 G1−G0 policy success，按 task/seed 聚类两个 draw、任务等权。20 个开发 scene 的方差不稳定，使用 0.10 方差下限；这是规划启发式，不是保证检出 10pp。正式推断照样报告 achieved clustered CI，不以“无显著”证明等效。

若输出 formal_scenes_per_task >100，在未开始任何 formal 模型结果之前，用相同 make_plan(tasks,20,n,2,20261120) 生成新 allocation。必须验证 development 数组逐项等于原开发计划，并重新验证新 test 未见、与 dev 完全不交。不同 n 会改变后续 task 的 test seed 切片，不能混用之前 100/task 的 manifest。子集规则依旧每 task scene_index<20，不按开发/正式效果挑 seed。

最低 n=100 时：600 units、1800 episodes、28200 最大请求、345600 broker API 上限（含感知调用，不能全称物理运动）。32768 output tokens × 28200 × $3.96/M = $3659.27 仅 output 的保守 peak 价格上界；输入没有全局 token cap，不能据此宣称完整账单上界。实际模型 usage + cache/offpeak 影响实际账单，现 cost_upper_usd 不是实付。

1800×2400 秒 = 50 天仅 episode budget 上界，环境初始化/归档/视频还有开销；实际平均时间可显著短，但不能保证 14 天覆盖。formal wall / cost 总限额应在完整开发后按实测均值和 p95、power-plan 与余量设置，不直接复用 development。request coverage 有强制 gate；wall/cost 仍需人工核对，耗尽的失败保留。

## 5. 正式 Snapshot 与 normal freeze

用 **不含开发 snapshot_manifest 的 preparation-config** 为最终 allocation 准备正式 Snapshot：

```bash
bash scripts/local/graph_experiment.sh prepare-snapshots \
  --config "$BASE/preparation-config.json" --plan "$BASE/final-allocation.json" \
  --split test --output "$BASE/formal-snapshots"
```

必须完整且 replay verified。n=100 时 completed=300。不要把 `$BASE/formal-snapshots/scenes.json` 回写到 preparation-config 后再 resume，签名会变；只给下一步新的 formal-config 引用完整 manifest。

formal-config 从 development/frozen.json 的 `config` 深拷贝，改 snapshot_manifest、acceptance_file、max_requests/cost_limit_usd/wall_seconds。保留模型、服务身份、七项runtime、全部每episode预算、提示词、源码集合、评分、graph_options、allowed_graph_ablations、mechanism_scene_subset、robotcalibration和assets。不要加入 initial_manifest 到 normal。

```bash
bash scripts/local/graph_experiment.sh freeze \
  --config "$BASE/formal-config.json" --plan "$BASE/final-allocation.json" \
  --development-root "$BASE/development" --output "$BASE/formal-normal"
bash scripts/local/graph_experiment.sh run --campaign "$BASE/formal-normal"
# 断点后的同目录继续：
# bash scripts/local/graph_experiment.sh resume --campaign "$BASE/formal-normal"
bash scripts/local/graph_experiment.sh analyze --campaign "$BASE/formal-normal" \
  --output "$BASE/formal-normal-analysis.json" --bootstrap 10000
```

如果 n=100 未变，将已验 allocation.json 字节复制到新的 final-allocation.json 即可；不要改既有 frozen plan。

## 6. 机制消融：全量来源、固定子集、直接配对

normal 全部固定 units 完成且 source 完整后：

```bash
bash scripts/local/graph_experiment.sh export-initials \
  --campaign "$BASE/formal-normal" --output "$BASE/shared-initials.json"
```

必须导出全部 600 units 的初始结果，包括不可用/请求失败，而非只导出成功任务或合法代码。目标消融运行时禁止初始生成 fallback。可用初始和 receipt/prompt/API/code/Snapshot hash 均验证并关联原 source，复用不二次收费。

每个 ablation-config 基于 normal frozen config：study=mechanism_ablation，groups=[G0,G1]，initial_manifest=shared-initials，graph_options 中只把预声明的一项置 false，faults=[none]；其他非实验变量不变。分别 diagnostic_slice / selective_invalidation / selective_recovery。固定 subset rule=first_scene_index,scenes_per_task=20 必须在最终开发前已经预声明于各配置。

```bash
bash scripts/local/graph_experiment.sh freeze \
  --config "$BASE/ablation-diagnostic-config.json" --plan "$BASE/final-allocation.json" \
  --development-root "$BASE/development" --output "$BASE/ablation-diagnostic"
bash scripts/local/graph_experiment.sh run --campaign "$BASE/ablation-diagnostic"
bash scripts/local/graph_experiment.sh contrast \
  --normal "$BASE/formal-normal" --ablation "$BASE/ablation-diagnostic" \
  --output "$BASE/contrast-diagnostic.json" --bootstrap 10000 --seed 0
```

其他两项同样流程。每项 120 units / 240 episodes，三项总 720；不是全 normal600 的三倍。contrast 直接比较同 subset 的 full G1−ablated G1，按场景聚类、双 draw 保留；不能比较各自“显著/不显著”来称机制贡献。三个次要 CI 未调整，不能声称 family-wise 95%。两侧全计划完成和所有四 hash 正确才 formal-ready。

## 7. 独立压力与最终读数

正式 stress 不与 normal/ablation 混合。当前全三任务共同适用 missing_baseline；support_refresh / target_shift 只适用有 place 的 Stack/Restack，Lift 上不能拿没触发的故障称恢复成功。最小压力计划应预声明独立固定 scenes；保留 assigned_fault 与 fault_fired，未达触发阶段的失败也保留，恢复率分母只按声明定义。

报告指标：G1−G0 policy 为主；G1−B0 共同 clean-return eligibility 的同原评分比较与异构额外技能完成契约区分。raw physical attainment 仅原物理状态达成诊断，controlled exit / budget / terminal 不计任务成功。另报 false allow/block 的标注覆盖和 unknown，不能用 final reward 反推逐次检测正确。

实际调用/费用在跨 normal/ablation 分析时按唯一 receipt 去重；每episode逻辑资源和归属成本另行报告。G0/G1 共享初始生成耗时在逻辑 deadline 公平扣除，但 wall/实际请求只发生一次，不把复制时间当再次实际付费。

## 8. 手工验收补充门槛

- 完整数目、所有失败、source_integrity_complete、uncertain requests/units、runtime_accounting_complete 都检查；自动分析并不替代账本完整性审计。
- started unit 如果只剩 pair receipt 而缺原 runtime_seconds，checkpoint 可显式标 recovered_runtime_unknown。这样的结果可保留物理结局，但不宣称端到端时间完整，正式耗时结论应标 unknown，不虚补时长。
- 每次 run/resume 重新验证 frozen bytes。服务进程更换或源修改可能阻断；不要为了继续把 frozen 哈希更新掉。
- 如最终 source/config 校验失败，先保留日志，修工程后重新建立合格最终开发，禁止悄悄消除 source_changed 历史。
- 磁盘容量以完整开发的 artifacts 实测字节/episode 外推 normal1800+ablations720+stress，并留足临时视频和 snapshot/archive 空间。每 campaign 全量归档一次，每 unit 仅业务代码与选中initial，避免按 unit 复制整批 normal 来源。


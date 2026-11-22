# 图反馈系统开发、冻结与正式评测计划

本报告基于正式目录源码及本地产物；未启动模型 API、正式评测或批量 Snapshot 仿真。当前只核实 Lift Seed 0 的三组端到端成功。Stack/Restack 的视觉几何与放置仍在开发，不能据此宣称三任务已验收或正式 GO。

## 实测成本与时间的依据

以下费用按当前请求账本的 V4 Pro 峰值、全部输入按未命中缓存计价，是保守用量估计，不是实际账单。共享初始请求只计一次；`reasoning_tokens` 已包含在输出 Token 中，不能再加一遍。

|产物|完整组别|结果|唯一请求|唯一输入/输出 Token|账本费用上界 USD|组内实际串行时间之和|
|---|---|---|---:|---:|---:|---:|
|`outputs/reproduction/graph-complete-20261120/deepseek-lift0-v4`|B0/G0/G1|3/3 成功；均无恢复|2|2,245 / 4,519|0.02085864|102.563 s|
|`outputs/reproduction/graph-complete-20261120/deepseek-restack21-v2`|G0/G1|0/2；各5次重规划，旧开发版本|11|160,234 / 59,738|0.44807136|991.222 s|

Lift 的 B0/G0/G1 实际时间为40.691/46.956/14.917秒。G1重用的初始生成约31.963秒，政策端到端时间46.880秒。排队资源估算应求实际执行时间或campaign active_seconds；策略延迟比较使用包含共享初始时间的 end_to_end_seconds。不能把三个组的逻辑 Token/费用直接相加为真实账单。

把该 Lift 单例机械外推至600个三组unit得到约17.09小时、12.52美元；这只表示简单无恢复路径的参考规模，不是三任务预测。把旧Restack两组单例外推600次已需165.20小时、268.84美元，尚未包含B0；它也不是当前版本均值。应等固定开发集完成后，按任务给出实际unit均值、p50/p95、初始化/存档额外时间及费用，再决定正式规模。禁止把新旧代码版本拼成正式效果样本。

## 固定分配与预算

一个unit是同一task/seed/repeat/fault下全部比较组，G0/G1共享一次初始生成，B0另一次。每组最多8个代码块，所以正常三组unit最多2+3×7=23个请求；两组压力或消融unit最多1+2×7=15个。

|正常分布计划|开发|正式默认|
|---|---:|---:|
|任务|3|3|
|独立场景/任务|20|100|
|每场景初始生成重复|2|2|
|unit / episode|120 / 360|600 / 1,800|
|完整请求覆盖上限|2,760|13,800|
|最多输出Token，32,768/请求|90,439,680|452,198,400|
|仅输出部分峰值费用 USD|358.1411|1,790.7057|
|每episode 2,400秒的总上限|10天|50天|

现有示例2400请求、500美元、7天不足以保证覆盖任一完整最坏路径。建议最终开发配置的累计安全上限设为2760请求、2000美元、15天；正式100场景配置设为13800请求、10000美元、90天。它们是可停止的保守资源许可，不是预计花费；通常实际消耗应低得多。单episode/模型预算保持一致，只放大累计上限。压力与消融各有独立账本和额外预算，不占用这13800次正常测试额度。

仍不能声称有限费用/时间“数学保证全部完成”：当前没有全局输入字节上限，CostLedger另要求两倍单请求预留余量；初始化、源码归档和视频编码不受episode超时完全约束。冻结前应以开发集最大prompt字节和各任务p95时间校验上述余量，并明确最终允许停止的资源线。正式冻结强制验证请求覆盖；触及费用/墙钟硬上限时保留未完成分配，不能报告已完成正式评测。

## 本轮补齐的最小工程入口

1. `prepare-development` 默认在创建目录前验证记录的运行环境、三个服务Schema、GraspNet Seed、递归源码身份，并实时比对服务；显式 `--exploratory` 才允许缺少来源的烟测，此类campaign不能用于正式开发验收。新服务清单可用 `outputs/reproduction/graph-complete-20261120/preflight-v3/service-inventory.json`，使用前仍会实时核对。NVML显存读数故障不冒充GPU已正常；现有服务可运行的证据与GPU监控可读性分别记录。
2. `prepare-snapshots` 复用已有 `vision_tasks.prepare_scene`，不启动模型。每个固定场景单独保存准备状态、replay验证和Snapshot文件hash；支持 `--max-scenes`、`--resume`。失败保留Seed，不按任务成功替换；unknown started不盲目重跑。完整正式清单必须在freeze前准备。
3. 正常开发配置可预声明 `allowed_graph_ablations`，列表中的每个字典只关闭一个图机制。机制消融可复用该正常开发验收，只允许命中声明的开关变化；模型、服务、源码、提示词、评分、单episode及模型预算保持一致，累计预算可随规模增长。B0不参与共享技能接口的故障注入或机制消融。
4. 先红后绿，新入口与现有campaign/durability回归 **66 passed**；证据为 `outputs/reproduction/graph-v2-evaluation-tests-20261120/preparation-{red,green-v1}.{xml,log}`。Shell语法与prepare-snapshots帮助入口通过。该测试没有真实付费或Snapshot批量仿真。

## 实际命令顺序

在项目根目录，先复制最终开发JSON为新配置，设置service_inventory及上述累计预算，并在首次prepare-development前声明三个消融字典。算法/提示词/服务最后修复必须发生在最终开发campaign前；后续任何源码漂移均会阻止续跑或正式冻结。

```bash
DEV="outputs/reproduction/graph_development_$(date +%Y%m%d_%H%M%S)"
bash scripts/local/graph_experiment.sh prepare-development \
  --config FINAL_DEVELOPMENT_CONFIG.json --output "$DEV" \
  --development-scenes 20 --test-scenes 100 --repeats 2
bash scripts/local/graph_experiment.sh run --campaign "$DEV" --max-units 3
bash scripts/local/graph_experiment.sh resume --campaign "$DEV"
bash scripts/local/graph_experiment.sh status --campaign "$DEV"
```

所有已实际初始化的模型失败都保留，不能以“合格成功数”驱动调度。不完整初始化/服务故障单列；先解决基础设施，不能在正式集上换Seed来筛成功。

开发完成后导出固定全量结果给power工具（仅本地文件处理）：

```bash
.venv/bin/python -B - "$DEV" <<'PY'
import json,sys
from pathlib import Path
from capx.experiments.graph_campaign import records
root=Path(sys.argv[1])
with (root/'development-records-v1.jsonl').open('x') as f:
    for row in records(root): f.write(json.dumps(row,allow_nan=False)+'\n')
PY
.venv/bin/python -B -m capx.experiments.graph_power \
  "$DEV/development-records-v1.jsonl" --config FINAL_DEVELOPMENT_CONFIG.json \
  --effect 0.10 --minimum-scenes 100 --repeats 2 --output "$DEV/power-plan-v1.json"
```

如开发波动要求大于100场景，必须在查看任何正式任务结果前扩大正式分配。现有API可生成新文件，无需修改开发集、复用开发Seed或改写原plan：

```bash
.venv/bin/python -B - "$DEV" <<'PY'
import json,sys
from pathlib import Path
from capx.experiments.graph_campaign import make_plan
root=Path(sys.argv[1]);old=json.loads((root/'plan.json').read_text())
n=json.loads((root/'power-plan-v1.json').read_text())['formal_scenes_per_task']
new=make_plan(old['tasks'],20,n,old['repeats'],old['allocation_seed'])
assert new['development']==old['development']
assert not ({x['seed'] for x in new['development']}&{x['seed'] for x in new['test']})
with (root/'formal-plan-v1.json').open('x') as f:json.dump(new,f,indent=2)
PY
SNAPS="outputs/reproduction/graph_formal_snapshots_$(date +%Y%m%d_%H%M%S)"
bash scripts/local/graph_experiment.sh prepare-snapshots \
  --config FINAL_DEVELOPMENT_CONFIG.json --plan "$DEV/formal-plan-v1.json" \
  --split test --output "$SNAPS"
# 中断/限量准备后的续跑：相同参数再加 --resume；unknown started需先核对。
```

正式配置指向上述完整 `$SNAPS/scenes.json`、真实三任务成功证据 `acceptance_file`、相同服务/运行配置，并把请求上限按 `3×n×2×23` 重算。保持最终开发的单episode和模型预算，不偷偷改变干预次数或推理参数。

```bash
FORMAL="outputs/model_comparison/graph_formal_$(date +%Y%m%d_%H%M%S)"
bash scripts/local/graph_experiment.sh freeze --config FORMAL_CONFIG.json \
  --plan "$DEV/formal-plan-v1.json" --development-root "$DEV" --output "$FORMAL"
bash scripts/local/graph_experiment.sh run --campaign "$FORMAL" --max-units 3
bash scripts/local/graph_experiment.sh resume --campaign "$FORMAL"
# 维护时从另一终端请求完整unit边界停止；不取消在途动作。
bash scripts/local/graph_experiment.sh request-stop --campaign "$FORMAL" --reason "计划维护"
bash scripts/local/graph_experiment.sh resume --campaign "$FORMAL"
bash scripts/local/graph_experiment.sh analyze --campaign "$FORMAL" \
  --bootstrap 10000 --output "$FORMAL/analysis-v1.json"
.venv/bin/python -B -m capx.experiments.graph_v2_analysis --run "$FORMAL" \
  --bootstrap 10000 --output "$FORMAL/offline-analysis-v1.json"
```

Campaign analyze负责完整分配/缺失判断；offline-analysis负责独立物理标签和全局共享请求去重账单。离线脚本单独输出的available-pair区间不能绕过campaign完整性门槛。所有输出文件使用新名字，避免覆盖旧实验。

## 冻结的统计方案

主估计为各任务等权的G1−G0政策成功率差，次要为G1−B0完整系统差异。每场景两次初始生成先平均，按task/seed抽整个场景簇进行分层bootstrap，10000次、固定随机种子；报告95%区间、场景及生成级胜/平/负、每任务成功率。300个独立场景不是600个独立重复；两个draw仅减少簇内模型生成噪声。不要把任务数当随机大总体推断。

100场景/任务对宏平均提升的检出能力取决于场景级配对差的方差。双侧5%、80% power近似下，方差0.10/0.25/0.50/1.00对应可检测幅度约5.1/8.1/11.4/16.2个百分点；希望检测10个百分点时约需27/66/131/262场景/任务。现有power工具以100为最低值，并使用0.10方差下限避免全平开发样本宣称零方差。20场景估计本身不精确，不能保证功效；报告实际区间及零效果也完全可接受。

原始物理reward、物理状态成功、政策成功分别保留。超时、受控退出、预算耗尽均为政策失败，不因最终静态物理状态偶然成功而提升计数。已发生的模型失败是分母中的零；缺失/未初始化场景单列，不伪造为成功或静默删除。正式固定分配没有完成时，仅报告描述性available-pair结果和缺失清单。

异常检测不能从reward反推。释放/控制/抓持分别用独立离线几何、FK和接触标签统计true_allow、false_allow、true_block、false_block及unknown；公开感知的conditional attachment/support不能当独立真值。错误放行比例是false_allow/(true_allow+false_allow)，不是经典FPR；误阻断比例同理。初始验证、重新取证、物理修正、LLM重新规划应拆开，不把每次感知都叫成功恢复。

正常、故障压力、单机制消融分开；压力记录是否真正触发，Lift无Place时支撑更新/目标扰动可能不适用，不能混成有效故障样本。先对已可恢复的故障取少量固定未见场景，再扩展；support_refresh/missing_baseline为证据机制压力，不自动代表现实滑落恢复。跨消融campaign目前不会共享同一次初始模型响应，只有各自G0/G1内部共享；因此跨消融差异不能宣称完全匹配初始代码的纯开关因果效应，需保持同场景并如实说明额外生成方差。

## 剩余硬条件

- Stack与Restack仍须取得真实公开视觉驱动的正常完成和有效物理恢复证据；Lift成功不能替代三任务验收。
- 最终开发必须来自同一冻结共同底座的20×2固定分配，当前历史调试Seed0/21不计入新场景。
- 缺失真实服务身份、加载源漂移、Snapshot不完整、修改了已冻结代码、未知在途请求/动作都会阻止自动推进；这些保护不是可以绕过的随机失败。
- 分析工具目前与执行源码一起列在来源清单，开发中更改分析源码也会产生漂移；因此应在最终开发前一起稳定，或未来明确版本化独立分析工具，不能事后偷改冻结清单。
- 请求预算可严格覆盖计划；完整费用/初始化耗时尚无数学上界。建议上述宽裕累计预算仍是有限资源上限，不等于必然花费，也不等于已承诺任务可靠性。

## 后续补齐：跨正常测试与消融的初始程序共享

本报告上文关于“跨消融campaign独立生成初始代码”的限制，在本次实现后可通过显式共享入口消除；不使用该入口的旧实验仍具有此限制，历史产物不被改写。

新增 `capx/experiments/shared_initial.py` 为纯来源校验模块。`export-initials --campaign NORMAL --output NEW_MANIFEST.json` 只导出完成的全部固定分配，包含初始无效程序和已知API失败，不看reward筛选。未知请求状态拒绝导出；未产生初始请求的场景保留为不可用条目，不能删除后宣称完整配对。正式消融不能使用开发初始程序。

消融配置的 `initial_manifest` 指向该文件。冻结和续跑核对原normal签名、固定unit、模型/参数、服务与源码、Snapshot、程序与Prompt/API文档、原始response/receipt及文件hash。`run_pair(initial_ref=...)`将通过校验的初始记录传给原有`run_episode(initial=...)`；新环境恢复后再执行原`check_pair`。缺失/错配会明确退出，不自动重新生成。B0及非none故障不能使用grounded normal初始。正式正常结果需在同一最终代码版本产生，不能用新实现冒充旧冻结版本。

复用仍扣每个策略的初始生成逻辑时间和请求次数，但新campaign不插入或发送一次初始收费请求；保留原canonical receipt路径，跨根离线分析去重。每个消融unit的14次可能新重规划仍可沿用15次保守逻辑请求覆盖。在线感知、Sandbox、monitor与物理评分未改。

先红后绿，新增测试覆盖来源篡改、场景/故障/模型参数错配、完整分配和失败保留、未知请求阻断、新环境Prompt重验、冻结与断点引用、跨根账单去重，以及无引用时旧Runner行为。联合回归 **122 passed**，日志：

- `outputs/reproduction/graph-v2-evaluation-tests-20261120/shared-initial-red.xml`、`.log`。
- `outputs/reproduction/graph-v2-evaluation-tests-20261120/shared-initial-green-v3.xml`、`.log`。

首轮新测试夹具缺少freeze要求的服务配置，已补显式fixture URL；未移除失败断言或降低运行来源校验。Shell语法与`export-initials --help`通过。所有该机制测试使用本地Mock，没有新增真实LLM请求，也没有声称正式消融效果已得到验证。真实命令及配置说明见 `docs/feedback/graph-v2.md` 最后一节。

# 图驱动执行反馈：开发与正式评测入口

V22 增量开发、条件图开关及小规模诊断命令见 [Graph V22](graph-v22.md)。下文保留 V21 历史流程；正式评测是否就绪以最新开发报告为准。

正式项目目录为 `/home/hanyu/dqb/cap-x-20261106`。此版本复用原有父进程 API broker、Bubblewrap actor、DeepSeek Provider、Snapshot 与独立物理评分。当前为 V21.3 开发验收阶段，DeepSeek 已恢复并完成新在线请求。V20 的 360 个 episode 已结束（包含 API 失败）；历史程序重放、新生成在线验收及不同版本结果分别报告。正式 1,800 episode 测试尚未开始，开发结果不能替代正式效果证据。最新状态见 [在线验收报告](../reports/graph-feedback-v21-online-resume-2026-11-22.md)。

## 三个系统

|系统|执行接口|诊断及恢复|
|---|---|---|
|B0|原始合法 Vision API|原程序级多轮反馈|
|G0|共同的有证据绑定的技能接口|相同验证器，保守重算未执行后缀|
|G1|与 G0 相同|按依赖图选择证据、失效传播及恢复范围|

G0/G1 共享初始 DeepSeek 程序、任务指令、Snapshot 和预算。后续恢复请求独立生成并记录。所有组沿用原物理成功判据；模型 FINISH、受控停止和验证器 PASS 都不能替代物理任务成功。B0 使用自己的合法原接口，故 G1−B0 表示完整系统差异，不能单独归因于图。

已接入 Stack、Lift、Restack 三种视觉任务，实际验收状态以最新报告为准；接入不等于任务通过。任务数不把 Seed、模式和改写重复计算。感知使用已有 SAM3、带 Seed 的 GraspNet 和 RGB-D，运动使用既有 PyRoKi；在线监控不接收仿真物体真值。独立物理状态只写入父进程离线诊断产物。

## 统一命令

在项目目录执行；脚本自动加载现有 `.env.local`、`.env.api.local`，不会输出密钥。服务地址、渲染环境、模型与预算来自显式 JSON 配置，先核对 `env_configs/graph_feedback/development-v20.json` 与实际服务。不负责启动或重启服务。

```bash
# 列出入口；各子命令 --help 不调用模型。
bash scripts/local/graph_experiment.sh --help

# 一对真实 DeepSeek 开发试验；必须使用新的输出目录，会产生 API 费用。
bash scripts/local/graph_experiment.sh pair \
  --config env_configs/graph_feedback/development-v20.json \
  --task restack --seed 21 --groups G0 G1 \
  --output "outputs/reproduction/graph_pair_$(date +%Y%m%d_%H%M%S)"

# 三系统正常分布开发试验。需要历史 Snapshot 时加 --snapshot FILE。
bash scripts/local/graph_experiment.sh pair \
  --config env_configs/graph_feedback/development-v20.json \
  --task lift --seed 0 --groups B0 G0 G1 \
  --output "outputs/reproduction/graph_lift_$(date +%Y%m%d_%H%M%S)"

# 以下V19压力配置已经准备；压力结果与正常分布结果分开统计。
# 压力试验必须单独记录，仅比较共享接口的 G0/G1。
bash scripts/local/graph_experiment.sh pair \
  --config outputs/reproduction/graph-complete-20261120/table-search-shape-v19/stress-support_refresh.json \
  --task stack --seed 0 --groups G0 G1 --fault support_refresh \
  --output "outputs/reproduction/graph_stress_$(date +%Y%m%d_%H%M%S)"
```

可选故障为 `none`、`support_refresh`、`missing_baseline`、`target_shift`。前两类是可重复的证据/版本机制压力，不等于自然发生的物理异常。`target_shift` 是显式的执行目标扰动；恢复仍必须经公开感知和机器人状态验证。故障和正常测试分开统计。三个图机制消融配置分别关闭诊断切片、选择性失效、选择性恢复，不更改验证器、接口或成功判据。

## 连续采样与断点

```bash
# 先在最终开发配置写入 service_inventory（graph_preflight 输出）。
# 默认验证服务来源与在线身份后才创建，不调用模型；仅烟测可加 --exploratory，
# 但该烟测 campaign 不能作为正式冻结的开发验收材料。
# 开发场景和正式场景全局互斥。
DEV="outputs/reproduction/graph_dev_$(date +%Y%m%d_%H%M%S)"
bash scripts/local/graph_experiment.sh prepare-development \
  --config env_configs/graph_feedback/development-v20.json --output "$DEV" \
  --development-scenes 20 --test-scenes 100 --repeats 2

# 按随机化组顺序串行执行。--max-units 是本次调度数量，不是成功样本目标。
bash scripts/local/graph_experiment.sh run --campaign "$DEV" --max-units 3
bash scripts/local/graph_experiment.sh status --campaign "$DEV"
bash scripts/local/graph_experiment.sh resume --campaign "$DEV" --max-units 3

# 不加 --max-units 会连续推进全部剩余 unit，仍受累计请求/费用/时间上限约束。
# 需要维护时从另一个终端请求在当前完整 pair 后停；不会取消在途动作。
bash scripts/local/graph_experiment.sh request-stop --campaign "$DEV" --reason "计划维护"
# 完成边界停止后，续跑自动消费旧 stop 请求，无需删除文件。
bash scripts/local/graph_experiment.sh resume --campaign "$DEV"
bash scripts/local/graph_experiment.sh analyze --campaign "$DEV" --output "$DEV/analysis-v1.json"
```

每个停止请求单独保存，不覆盖或删除历史请求。`resume` 只推进尚未开始的完整 unit；已完成但失败的结果也保留。检查点不得删减计划场景或改变组顺序，结果缺组会拒绝标记完成。只有全部固定分配完成、没有未知请求且配对来源完整，才可能报告正式评测就绪；部分结果仅作描述。

失败也标记为已完成，不重新抽样替换。一个调度 unit 包含一个任务、场景、初始生成重复和故障条件下的全部组。账本锁防止两个进程同时执行同一 campaign；请求在发送前保留预算，完成响应按 prompt hash 缓存。若请求是否完成不明，或物理 unit 中途退出且缺少完整 `pair-result.json`，自动续跑会拒绝，需先核对实际请求与动作，不能盲目重放。

正式开发开始前，应在配置内设置 `service_inventory`，指向本版本 `graph_preflight` 的递归服务源码清单。可用 `.venv/bin/python -B -m capx.experiments.graph_preflight --config CONFIG.json --output NEW_INVENTORY_DIR` 创建；此命令仅 GET Schema/健康状态与读取进程元数据，不调用模型或感知推理。若计划后续图机制消融，在开发配置中预声明 `allowed_graph_ablations` 为三个完整开关字典（分别只有 `diagnostic_slice`、`selective_invalidation`、`selective_recovery` 之一为 false）。正式消融仅允许名单内的单开关变化，其余单 episode／模型预算、共同底座、服务、模型、评分和提示词不能变化；总请求数、总费用和总墙钟可随规模增长。

```bash
# 开发结束、算法冻结后，按已分配正式场景准备初始 Snapshot（无 LLM 请求）。
# 这里只验收初态稳定及可重放，不运行任务，也不按任务成功选择场景。
SNAPS="outputs/reproduction/graph_formal_snapshots_$(date +%Y%m%d_%H%M%S)"
bash scripts/local/graph_experiment.sh prepare-snapshots \
  --config FINAL_DEVELOPMENT_CONFIG.json --plan "$DEV/plan.json" \
  --split test --output "$SNAPS"
# 可选 --max-scenes 限制本次准备数；续跑显式 --resume。
bash scripts/local/graph_experiment.sh prepare-snapshots \
  --config FINAL_DEVELOPMENT_CONFIG.json --plan "$DEV/plan.json" \
  --split test --output "$SNAPS" --resume

# FORMAL_CONFIG.json 指向 $SNAPS/scenes.json、任务 acceptance_file、同一服务清单；
# 总预算按实际计划设置，per-episode/model 预算沿用最终开发。
FORMAL="outputs/model_comparison/graph_formal_$(date +%Y%m%d_%H%M%S)"
bash scripts/local/graph_experiment.sh freeze \
  --config FORMAL_CONFIG.json --plan "$DEV/plan.json" \
  --development-root "$DEV" --output "$FORMAL"
bash scripts/local/graph_experiment.sh run --campaign "$FORMAL" --max-units 3
bash scripts/local/graph_experiment.sh resume --campaign "$FORMAL"
```

Snapshot 准备也不覆盖旧目录。失败保留原 Seed，不自动重抽或重试；未知 started 状态需核对后处理。清单不完整时正式 freeze 会拒绝。正常测试、压力测试和消融应各自独立 campaign；已有失败不得混入新算法版本后宣称同一冻结实验。

正式冻结要求开发验收材料，不能只把上述 development 配置改名后启动。`acceptance_file` 的 `evidence_files` 必须直接指向最终开发 campaign 账本内、已经通过来源校验的 `episodes/G0/result.json` 或 `episodes/G1/result.json`；每个任务至少有一个真实 DeepSeek + 在线 Vision 的物理与执行策略成功。独立 Oracle、复制到外部的成功 JSON、只有 B0 成功、预算/受控终止而仅保留 reward=1，都不能作为该验收。所有失败开发尝试仍保留并计入场景尝试数量，不按成功筛选开发场景。需保留完整新场景 Snapshot 清单、实际服务/运行环境清单、代码/提示词/评分来源、任务真实成功证据及每任务至少 20 个完整独立开发场景。样本量由开发集场景级波动、目标提升与时间费用估计确定；正常正式比较包含三个系统。正式集不用于继续调参。正式 normal freeze 还会按实际 unit、组和 `max_blocks` 强制检查请求上限覆盖完整计划。三任务 × 100 场景 × 2 初始生成 × 3 系统、16 blocks 的请求上限至少为 28,200（G0/G1 初始共享，只计一次）；开发配置的小额上限不会自动放大。费用及总耗时须另外估计，不能用请求数推断全费用有保证。

```bash
# 使用实际运行配置进行样本量/预算规划；此命令不调用 API。
.venv/bin/python -B -m capx.experiments.graph_power DEVELOPMENT_RECORDS.jsonl \
  --config env_configs/graph_feedback/development-v20.json --output NEW_POWER_PLAN.json
```

## 执行图与观测边界

```mermaid
flowchart LR
  rgb[版本化公开 RGB-D] --> obj[物体估计/几何先验]
  robot[当前 TCP/关节] --> plan[带来源 ID 的技能计划]
  obj --> plan
  phase[抓持阶段与剩余预算] --> gate[边界前置条件]
  plan --> gate
  gate --> call[父进程真实 API 调用]
  call --> check[公开状态验证]
  check --> next[继续/局部恢复/重新规划/终止]
  check --> diag[图反向诊断证据]
  diag --> llm[DeepSeek 恢复程序]
  llm --> plan
```

节点记录程序片段、带版本的观测/估计、计划、调用、验证和历史事实；边区分数据依赖与物理前置条件。历史调用只说明已经发生，不会因失效变回待执行动作。新 actor 丢弃 Python 变量，但持物阶段、已完成动作、失效信息、剩余 API/局部恢复预算持续保留。关键目标必须由父进程用真实来源 ID 计算，actor 不能用任意 XYZ 冒充有依赖的放置目标。

运动调用是阻塞接口，**仅支持 API/技能边界干预，不能声称运动中取消**。公开几何带有可见性和刚体假设；同一 RGB-D 估计的重复测量不构成独立证据。释放要求当前证据，历史正确定位不能在可能发生接触后继续充当释放许可。

## 产物与统计

每对试验保留初始程序、完整请求与响应、用量凭据、API 文档、源码副本、Snapshot hash、各 block、图事件、公开观测 NPZ、前后物理 Trace、结果与视频/仿真时间戳。密钥和鉴权头不写日志。

场景是独立统计单位，同一场景的两次模型初始生成属于簇内重复。主比较为 G1−G0，附 G1−B0；报告配对胜负、按场景聚类的置信区间，以及失败、受控退出和环境初始化缺失。错误放行/误阻断由独立离线状态核验；没有可判定标签时保留未知，不能由最终 reward 反推检测准确性。

费用字段优先是实际 token × 已记录价目得到的估计或上界，不是账单。共享初始请求按全局请求账本去重；模型/服务超时和未收到用量的请求单独保留，不计成零成本。

## 正常测试与图机制消融共享初始程序

在最终开发配置中提前声明三个允许的单机制开关，并在 normal 与三个消融配置中保持相同的固定子集规则：

```json
"mechanism_scene_subset": {"rule": "first_scene_index", "scenes_per_task": 20}
```

规则选择最终冻结正式 plan 中每任务 `scene_index` 0–19，保留全部初始生成重复；不接受自选 Seed、成功率排序或跳过失败。配置在开发冻结后不能添加、修改或删除。normal 仍执行每任务全部 100 个场景。若预实验功效分析要求扩大正式场景数，应在读取正式结果前生成并冻结扩大的完整 plan；提前声明的前 20 场景规则不变。正常三组 campaign 完成后，以下无模型命令会导出**所有固定unit**的初始程序来源；不读取reward筛选成功，截断/无效程序和已有API错误也保留。存在未知请求完成状态时拒绝导出。未初始化、根本没有初始请求的unit会以不可用项保留，不能静默删掉后做完整配对消融。

```bash
bash scripts/local/graph_experiment.sh export-initials \
  --campaign "$FORMAL" --output "$FORMAL/shared-initials-v1.json"
```

消融配置新增 `"initial_manifest": "/absolute/path/to/shared-initials-v1.json"`，`study` 使用 `mechanism_ablation`，`groups` 使用 `["G0", "G1"]`，`faults` 保持 `["none"]`。沿用正常测试的完整 plan、Snapshot、模型与单 episode 预算，仅更改预声明的一个图开关及累计资源额度，再通过既有 `freeze/run/resume` 命令执行；调度只执行预声明的 20 场景子集。正式 `mechanism_ablation` 缺少有效 `initial_manifest` 会立即拒绝冻结，不能静默独立生成。正式消融只能引用正式normal的初始结果，不能借用开发场景。共享初始机制必须在最终开发和正式normal冻结前就已加入代码；不能拿旧版本产物跳过来源校验。

冻结时校验源清单恰好覆盖 normal 完整分配，目标恰好覆盖预声明子集，且两者的完整 plan 一致；同时校验 normal 签名、真实单次 DeepSeek 请求凭据、prompt/response/receipt 原始文件、源码和服务身份、模型参数及四个配对 hash。非选中场景的不可用初始仍留在源清单；选中场景不可用则拒绝执行，不能换场景或重新生成。执行时恢复Snapshot后仍重新核对当前Prompt/API文档。不匹配会明确停止，不会生成一份新初始代码替代。`run_pair(initial_ref=...)`复用既有`run_episode(initial=...)`，没有新增Agent或改变Sandbox。

三任务 × 20 场景 × 2 次生成 × 2 系统，每项消融为 240 episodes，三个单机制消融合计 720；加 normal 的 1,800 为 2,520 episodes。每项消融有 120 units；在当前16代码块预算下，保守请求覆盖上限为3,720，共享初始后最多新增3,600个恢复请求。全部normal加三消融最多新增39,000个请求，分别配置的保守请求上限合计为39,360。压力测试使用独立提前分配的适用场景，不混入正常或机制消融统计。

初始请求仍属于各episode的逻辑资源消耗，原生成耗时照常从每个episode期限扣除；新campaign账本不再发送或收费一次初始请求。当前16代码块预算下，每个两组消融unit最多30个新的重规划请求，31请求的保守覆盖不必缩减。结果同时保存 `initial_reused` 和 `initial_source`；使用多个 `--run` 的离线分析会按原始canonical receipt/response ID去重，不能把各episode的逻辑费用简单相加。

开发调试可显式指定同一manifest中的单个初始：

```bash
bash scripts/local/graph_experiment.sh pair \
  --config ABLATION_CONFIG.json --task stack --seed FIXED_SEED --repeat 0 \
  --groups G0 G1 --snapshot SAME_RAW_SNAPSHOT.json \
  --initial-manifest SOURCE_NORMAL/shared-initials-v1.json --output NEW_OUTPUT_DIR
```

共享仅适用于正常 `none` 故障到同场景的单机制消融；B0使用不同合法接口，不能导入grounded初始。源请求文件与账本均不改写，新输出独立保存引用证据。

完成某个机制消融后，直接比较 **完整 G1 与该消融 G1**；两边使用上面提前声明的相同子集和初始程序。不能用两个 campaign 各自 G1−G0 的显著性差，替代该机制的贡献：

```bash
bash scripts/local/graph_experiment.sh contrast \
  --normal "$FORMAL" --ablation "$ABLATION_CAMPAIGN" \
  --output NEW_MECHANISM_CONTRAST.json --bootstrap 10000 --seed 0
```

该命令不发请求。它校验两个正式 campaign、共同底座、原始 initial 来源、预声明子集及全部重复，复用按任务内场景整簇重采样的统计实现。主结果为相同 grounded 完成契约下的策略成功率差；原始物理状态达成另列诊断，受控退出或预算耗尽不能计作任务成功。缺失结果、未知请求或未完成分配只产生描述性结果。三个预声明机制对比各给未校正的 95% CI，不声称全族同时覆盖。

归档按层级保留：campaign 冻结目录保存完整代码、服务/资产、Snapshot 清单及 shared initial 来源字节一次；每个 unit 保存业务依赖、当前初始程序/请求凭据/配对结果与自己的 Snapshot，通过 `campaign-reference.json` 关联完整冻结目录，避免每个 unit 复制全部正常实验。每次 `run/resume` 重新全量验证；执行中仍检查当前业务代码、选中请求字节和冻结签名。缓存仅存在当前调用内，下一次续跑重新校验；不会跨调用信任缓存。保留源 campaign 与冻结目录，不能只搬走 unit 后删除它们。CLI 独立 `pair` 仍严格检查共享来源，不存在校验失败后重新请求的后备路径。

当前v20开发配置为16代码块、192实际API、24局部恢复、8视角调整、2400秒episode上限；旧development.json保留8块用于历史复跑。新增完整持物侧面与当前底边的公开几何证据，仍是同相机条件推断；桌面一像素语义边界及2mm首表面阈值未放宽。完整测试、失败与真实验收进度见 `../reports/graph-feedback-resume-2026-11-21.md`。

V17共同底座还将被阻断的桌面位置登记为搜索历史，重新选择可验证区域；空夹爪后清除当前抓持身份，并将旧身份保存在last_grasp_loss历史字段。G0/G1使用同一实现与预算。V16在线开发因已确认状态错误人工中断，不能算作完成的三组配对。

V20在技能入口及每次真实物理API前检查当前公开夹爪状态。已确认空夹爪会阻断尚未执行的动作，清除当前抓持身份并保留历史与预算；缺失夹爪证据要求重新读取。正常夹爪开度本身不构成抓持成功证据，既有几何与共动验证仍然必需。此共同修复同时用于G0/G1。

V20固定开发批次已完成，目录`outputs/model_comparison/graph_final_v20_20261122/development`，60个新Snapshot已验收，120个unit/360个episodes均已结束，包含API失败，不能视作360次有效控制实验。只读进度命令：`bash scripts/local/graph_experiment.sh status --campaign outputs/model_comparison/graph_final_v20_20261122/development`。有运行中的runner时不要同时resume。正式1800个episodes尚未启动。详情见`../reports/graph-feedback-v20-development-2026-11-22.md`。

## V21 动作准入与有界恢复

以下保留V21.1的机制说明，当前V21.3另包含桌面恢复、原生horizon终止与响应审计修复；以每次运行源码归档为准。历史 V20 结果、V21 首次失败、V21.1 复跑分开保存。当天早期请求曾返回HTTP402，此后已恢复真实API验收；下述历史程序重放仍不得当作新模型闭环验收，正式评测仍未就绪。详见 [验收报告](../reports/graph-feedback-v21-2026-11-22.md)。

共同底座修复：历史 `verification_fact` 只是授权发生过的记录。释放前最后观测后，原许可必须仍有效，并用当前持物/支撑重新验证；桌面验证内部再次捕获的画面也必须同步到持物证据。多段 place 的来源更新立即阻断后缀；完全缺少支撑证据记 UNCERTAIN。仅有与历史一致的部分可见点时，只有 ascend/above_target 阶段、当前与目标持物底面均高于支撑至少 50mm、非下降运动可以条件性继续。下降/释放不能使用该历史许可。这是静态支撑假设下的启发式边界检查，不是动态避碰或独立传感器验证。

`GroundedRecovery` 复用既有技能并生成下一 actor 的完整代码，仍经 Sandbox、broker、相同验证器和预算执行。支持支撑更新后重新放置、释放后新撤退、抓持验证、抓持丢失后重建场景与受约束重抓。动作次序由显式技能配方决定；图只决定证据和重算范围。G0 重新取得未执行后缀所需来源；G1 只复用具有正确版本键、坐标系、当前 epoch 且有效的 API 观测句柄，释放检查摘要不属于可复用句柄。已完成调用保留为事实，新计划有新 ID；未知物理效果不自动恢复。

配置 `env_configs/graph_feedback/development-v21-3.json` 的 budgets 新增：

- `bounded_recovery: true`：G0/G1 共同开启；设为 false 保留原 LLM 多轮恢复路径。
- `max_recovery_plans: 6`、`max_recovery_no_progress: 2`：每 episode 自动恢复配方上限、同一阶段/对象无进展上限。每份配方预留一次既有 local attempt，不能跨 actor 刷新额度。
- 原有 16 代码块、192 实际 API、24 局部尝试、8 视角、2400 秒、32768 输出 Token 不变。超出覆盖范围才进入原 DeepSeek 恢复入口，仍受原全局/episode 上限约束。

`bounded_recovery_events` 分别记录保留/选择来源和新计划**实际使用并返回**的来源；未执行计划不计复用收益。`contract_calls`、原 `actual_api_calls`、观测 NPZ 数、工具日志、`component_timing` 用于区分获取证据、控制、监控与 LLM 成本。`avoided_acquisition_apis` 是相对同配方重新获取的反事实读数，最终节省量仍用配对实测核对。工具服务时间是嵌套耗时，不可和控制/监控桶重复相加；历史程序重放的 `llm_seconds` 含原初始生成的配对预算时间，不是本轮 API 耗时。

当前真实模型恢复验收（DeepSeek已恢复；使用V21.3与新目录，防止覆盖）：

```bash
cd /home/hanyu/dqb/cap-x-20261106
bash scripts/local/graph_experiment.sh pair \
  --config env_configs/graph_feedback/development-v21-3.json \
  --task stack --seed 953265258 --groups B0 G0 G1 \
  --snapshot outputs/model_comparison/graph_final_v20_20261122/development-snapshots/scenes/stack-953265258/prepared-snapshot.json \
  --output "outputs/model_comparison/graph_v21_smoke_$(date +%Y%m%d_%H%M%S)"
```

零新增模型调用的诊断命令（历史 DeepSeek 初始程序 + **真实在线** Vision，局部恢复无法处理时退出，不查询 LLM）：

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -B scripts/local/replay_graph_initial.py \
  --config env_configs/graph_feedback/development-v21-3.json \
  --source-unit outputs/model_comparison/graph_final_v20_20261122/development/units/7ec0118521e1e2a5ea2c \
  --snapshot outputs/model_comparison/graph_final_v20_20261122/development-snapshots/scenes/stack-953265258/prepared-snapshot.json \
  --fault support_refresh \
  --output "outputs/reproduction/graph_v21_diagnostic_$(date +%Y%m%d_%H%M%S)"
```

`--fault` 可选 none/support_refresh/missing_baseline/target_shift。输出 `provenance.json` 标注历史模型来源；校验原响应、代码 hash、HTTP 200 凭据以及配对 Prompt/API/Snapshot hash。该入口不生成正式 campaign、不能代替新模型重复。

通过真实小规模验收后，复用既有固定开发分配与 60 个 Snapshot，而非重新抽取 Seed。以下仅准备新版本开发冻结，不会调用模型；正式规模仍由新开发结果决定：

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
DEV="outputs/model_comparison/graph_v21_dev_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python -B - "$DEV" <<'PY'
import json,sys
from pathlib import Path
from capx.experiments.graph_campaign import freeze_campaign,validate_development_services
c=json.loads(Path('env_configs/graph_feedback/development-v21-3.json').read_text())
old=Path('outputs/model_comparison/graph_final_v20_20261122')
c['snapshot_manifest']=str((old/'development-snapshots/scenes.json').resolve())
plan=json.loads((old/'allocation.json').read_text())
validate_development_services(c)
freeze_campaign(Path(sys.argv[1]),c,plan,phase='development')
PY
# 恢复 API 后，按完整配对 unit 续跑；首次先验收两个 unit
bash scripts/local/graph_experiment.sh run --campaign "$DEV" --max-units 2
bash scripts/local/graph_experiment.sh resume --campaign "$DEV"
```

正式效果与图机制消融继续使用本文既有 freeze/export-initials/contrast 流程，不能将 V20 初始来源清单当作 V21 正式共享来源；版本、共同底座及预算校验保持不变。

本轮恢复验收范围必须区分：正常完成、支撑版本更新、目标偏移、缺失基线有历史模型程序＋真实 Vision 诊断；抓持丢失后的重抓，以及单独“已释放但撤退过期”目前只有软件/Mock 回归，尚不能声明在线物理恢复已验收。所有新生成和 LLM 重新规划都需要 DeepSeek 402 解除后继续验证。

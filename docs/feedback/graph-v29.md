# V29 自动因果构图与局部计划修复

在现有 ContractGraph、GraphExecutive、ConditionPlanner、父进程 broker 和 Sandbox 上增量实现。沿用 V28 的精确输入缓存、共享感知、状态记忆和代价感知补证。没有安装 LOGIV、Fast Downward、VAL 或新模型。

## 实际入口

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN="$PWD/outputs/model_comparison/v29_$(date +%Y%m%d_%H%M%S)"
IK_RUNTIME="$PWD/outputs/model_comparison/graph_v28_20261124_155408/ik-runtime.json"
.venv/bin/python -B -m capx.experiments.contract_models \
  --config env_configs/graph_feedback/development-v29.json \
  --ik-runtime "$IK_RUNTIME" --seed 0 --study evidence_expiry \
  --ablations no_causal no_local_repair no_resume_validation no_cost \
  --v28-control --output "$RUN/evidence0"
```

此命令实际调用 DeepSeek V4 Pro 和现有 SAM3 / GraspNet / IK，运行 7 个配对 episode。每个单元只有一份新 DeepSeek 初始程序，各组严格共享完整 Snapshot、Prompt、技能和评分。恢复只在必要时追加真实模型请求。当前 IK runtime 文件必须对应在线服务；重启后应按 [V28 服务说明](graph-v28.md) 使用原 CPU IK 启动脚本生成新 runtime 文件。URL 来自配置和 runtime 参数，核心代码不写死端口。

其余固定开发单元，可在同一 shell 依次运行；每个输出目录必须不存在：

```bash
.venv/bin/python -B -m capx.experiments.contract_models --config env_configs/graph_feedback/development-v29.json --ik-runtime "$IK_RUNTIME" --seed 0 --study normal --v28-control --output "$RUN/normal0"
.venv/bin/python -B -m capx.experiments.contract_models --config env_configs/graph_feedback/development-v29.json --ik-runtime "$IK_RUNTIME" --seed 0 --study effect_displacement --output "$RUN/physical0"
.venv/bin/python -B -m capx.experiments.contract_models --config env_configs/graph_feedback/development-v29.json --ik-runtime "$IK_RUNTIME" --seed 21 --study normal --output "$RUN/normal21"
```

四个单元共 14 个 episode，属于开发诊断，不是正式统计评测。不要覆盖未完成单元；检查结果后选择新目录。任务仍为现有 PickPlace，不新增仿真环境。

## 机制和约束

- 初始模型程序采用可检查的子集：无条件的 literal intent ID 调用，或只读常量列表/元组上的单层有限 `for`。编译器只展开已证明有界、循环体仅含该次 Skill 调用的序列；拒绝动态迭代、分支、变量重赋值、循环体修改和 `for … else`，不做任意 Python 静态分析。
- 可信技能库声明 `requires`、`requires_false`、`effects`、`deletes`、`invalidates`、`data_requires`、`resources` 和有限参数绑定。模型不能提交自己的前置条件、效果或动作许可。
- 编译模型实际调用顺序，自动建立因果支持、证据义务、数据、冲突及资源边。必要的显式 `after` 仍受检查；V29 的区域搬运定义不再预写抓取—放置 `after`。
- 预期效果只用于计划推演。真实条件只能由父进程 verifier 建立，预测删除也不能伪造 FALSE。负条件必须得到明确 FALSE，UNKNOWN 不按闭世界假设处理。
- 每次实际观测版本、返回调用、停止调用、未来计划与局部修改均进入原 DependencyGraph。旧版本和已完成调用保留。未知物理效果继续隔离。
- 修复先定位受影响动作与必须替换动作，回归保留后缀的条件，从参数化技能库进行有界代价搜索，再验证修复片段＋剩余计划。提交时检查证据版本未变、资源无在途占用，并给新动作分配新 ID。
- 搜索最多 512 次状态展开、深度 6、3 次范围扩展。新物理修复计入原 repair / skill / API / episode 预算，不另开无限恢复额度。
- 未来才消费的工具证据可以延期获取，仍在完整计划验证中列为义务；这不会把 UNKNOWN 写成当前 TRUE。最终每次派发仍经过 broker 的实时验证及技能内原有安全检查。
- 常规补证及已验证计划的续接由确定性父进程生成单 Skill 片段，重新进入相同 Sandbox / broker。`actor-*-program.json` 区分 `real_deepseek` 与 `deterministic_verified_plan`，后一类不算新的模型输出。

## 对照含义

| 组 | 恢复策略 | 共同能力 |
|---|---|---|
| G0 | 强保守后缀补证；顺序计划回归和相同技能库搜索 | 当前感知、记忆、确定性执行、安全准入与预算 |
| G1 | 自动因果支持、选择性受影响范围、局部修复和续接验证 | 同上 |
| no_causal | 不推导自动因果/数据支持；保守处理剩余范围 | 安全检查和技能不变 |
| no_local_repair | 关闭确定性局部修复与自动续接，沿用现有条件规划＋模型恢复 | 感知和安全不变 |
| no_resume_validation | 不要求搜索保持整个后缀回归条件，不验证拼接兼容性 | 每次物理派发的实时安全许可仍保留 |
| no_cost | 不应用 V28 共享需求和生命周期代价排序 | 精确缓存与实际检查仍保留 |
| V28-G1 | V28 条件规划和模型恢复策略 | 使用当前共同契约和技能，**不是历史源码检出** |

`no_local_repair` 测量的是局部修复＋确定性续接整项能力，不是纯拓扑贡献。精确机制反例证明续接验证开关不同；如果线上场景没有触发相应分支，不能宣称该消融产生了任务效果。

每单元上限 24 次模型请求、项目估价 US$5、7200 秒；每 episode 24 Skill、256 API、1200 秒、12 个代码块、12 次补证、6 次修复。DeepSeek V4 Pro 使用 low reasoning effort 和 32768 输出 token 上限。

物理扰动的机制对照可把首条命令的 `--study evidence_expiry` 改为 `--study effect_displacement`，保留 `no_causal no_local_repair --v28-control`，使用新的输出目录。

## 验证与分析

```bash
.venv/bin/python -B -m pytest -q tests/graph_v2/test_v29_causal_repair.py
.venv/bin/python -B scripts/local/causal_mechanism_gate.py --output "$RUN/symbolic"
.venv/bin/python -B scripts/local/analyze_contract_models.py "$RUN/evidence0" --output "$RUN/contracts.json"
.venv/bin/python -B scripts/local/analyze_perception_cost.py "$RUN/evidence0" --output "$RUN/perception.json"
```

`causal_mechanism_gate` 是**精确符号状态夹具**：没有仿真、Vision 或 LLM，不能混入物理成功率。它检验支撑失效后组合“暂放—修复—重抓”并续接独立分支。真实实验结果见 [V29 报告](../reports/graph-feedback-v29-development-2026-11-24.md)。

参考 [LOGIV 固定提交](https://github.com/Enqiao-Lu/LOGIV/tree/0ea7a7ff765016a33409c54572d733bed4503d4d/logiv/core)：借鉴其 live support、affected/replaced 区分、条件回归及 validate-before-splice 思路。这里是适配 CaP-X 三值证据与现有 broker 的实现，不声称等价于 LOGIV 的 PDDL/VAL 验证或复现其论文效果。

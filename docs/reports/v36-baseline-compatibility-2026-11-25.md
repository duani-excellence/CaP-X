# V36 外部 Baseline 兼容性与原始 CaP-X 最小任务适配

正式工程 `/home/hanyu/dqb/cap-x-20261106`，分支 `feature/graph-complete-20261120`，HEAD `8052fa7ed3d2d74955a82454d7242a41dc0c9307`。保留原有未提交工作。结果位于 `outputs/model_comparison/graph_v36_20261125_175831`，本文实验没有覆盖历史结果。

**结论：本次原始程序策略最小适配完成真实 DeepSeek + Vision 物理执行，但 PickPlace 为 0/4。它没有被新增图准入挡住，仍因目标识别和抓取验证不足而失败。后续系统确实也有独立的接口回归，不能把所有失败归因于原始能力，或反过来全部归因于新增约束。** 单场景、不同工具/恢复预算只适合诊断，不能据此给方法排位。

## 版本与实际策略

| 方法 | 实际实现 | 本轮兼容性与范围 |
|---|---|---|
| 原始 CaP-X / B0 最小任务适配 | 原 `graph_v2` B0 的完整 Python 执行、`_build_multi_turn_decision_prompt` 程序级 stdout/stderr 反馈；新 `native_b0.py` 仅适配原生 PickPlace、公开任务目标、Snapshot、原生评分和 API 生命周期 | 新 DeepSeek 2 次，原始公开 `get_object_pose/sample_grasp_pose/goto_pose/open/close/home`。不安装 Contract Skill、Graph Executive、Persistent Harness 或确定性恢复。保留 P0 Sandbox、父进程预算、标定与 BoundedIK。**非官方原版逐字复现；不是 CaP-Agent0** |
| G0 | 原强保守恢复 + 同一共同 Skill/Broker；未缩减技能库、验证器或预算 | V36 open 模式真实模型主试验见总报告；本次另外用 legacy/v30 兼容路径重放，以便与 LOGIV 保持共同能力 |
| G1 | 同一共同 Skill/Broker，条件回归、选择性恢复与局部修复 | 保留原策略；没有将 B0/LOGIV 接到此 Harness，也未给 G1 加任务专用动作分支 |
| LOGIV 共同底座适配 | 官方 revision `0ea7a7ff765016a33409c54572d733bed4503d4d`，7 个核心/规划文件与 Git 字节一致。真正导入其执行器、Verifier、因果图、Repairer | 共同 Vision Skill 替代 VLA/GPT-4o；有限技能搜索替代 FastDownward；符号兼容验证替代 VAL。本轮真实 Vision 重放。**不是原论文原版评测** |
| ASPIRE 官方原版/完整适配 | 官方 revision `fa00f0ef2bc1d91d594b9aa502c22133c9f538f7`；已复用部分感知/规划服务协议 | 完整 coding-agent 技能发现、技能库准备、训练/开发与评测分割、调用成本记录未接入统一 Runner。当前本地 ASPIRE 另有 16 条未提交状态，未修改。其自带 agent/session 不能直接冒充论文复现；本轮未运行策略 |
| CaP-Agent0 官方原版/完整适配 | 官方 revision `acac62fcb1b6f9e02975004a2c729a65e11bffed`；从已有部分 checkout 的 Git 对象核查原配置与调用逻辑 | 图像差分、ReducedSkillLibrary、技能合成、并行多模型集成未接入共同计量与安全边界。纯文本 DeepSeek 单程序重试不能冒充该方法。尚不可参赛，不记零分 |

核查复用 V30 `upstream-lock.json` 与既有官方 checkout；`external-source-audit.json` 保存当前 revision/校验，`baseline-source-audit` 保存核查的官方 Agent0 原始字节。未重新下载、另搭 Runner、改上游源码或升级环境。LIBERO-Recovery 保持此前协议/策略适配阻塞，未把环境重置计为方法成功。

LOGIV 当前合法输入是字面量 `execute_skill` 计划，不支持任意开放式 Python、F/T 或 Persistent Harness。V36 语义注册和 Decision State 不会静默套给它；统一入口现在拒绝把含 open-runtime 能力的配置直接作为 legacy/LOGIV 配置执行。B0 PickPlace 必须独立生成自己的合法原始 API 程序，不能重用 G0/G1 Skill 代码。

## 新 DeepSeek B0 最小实验

`b0-minimal-seed0`：与 V34/V36 同一 Seed 0 Snapshot，初始物理 hash `642d1d5a14b89950ac7941358992a46846002a685018a1f9e56a180af9333184`。相同原生完整成功判定；Vision-only，不把 F/T 或真值加入原始策略。上限 2 请求、2 完整程序、256 API、1200 秒 episode；模型 `deepseek-v4-pro`，输出预算 32768，reasoning low。两次真实 HTTP 200，finish=stop，无重试。

1. 新初始程序含 `__name__` 和私有函数名，Sandbox 在动作前拒绝。0 物理调用。该错误确实通过原程序反馈发送给模型，没有人工修剪程序。
2. 新恢复程序修正 Python 写法，执行完整四轮抓取/搬运/释放，返回 ok，但原生评分 **0/4、reward=0**。原始程序没有逐物体成功验证；本次两次请求中的恢复用于修正代码，因此**尚未测试收到该物理失败后的下一轮模型恢复**。

实际调用：43 API（16 goto_pose、12 get_object_pose、5 open、4 grasp sample、4 close、2 home），SAM3 16、GraspNet 4、IK 44。端到端 282.07 秒，模型 HTTP 合计 244.10 秒；第二个程序执行约 33.03 秒。低调用数不能当效率优势，因为任务进展为 0。

公开工具与私有离线评价证据分开保存在 `b0-tool-diagnosis.json`、`b0-minimal-offline-diagnosis.json`：

- 模型选择 `bin2` 结果作为目标箱。其 SAM3 分数 **0.03345**，返回中心距同帧 Milk 定位 **0.628 mm**，尺寸仅约 **61×83×155 mm**。原始 API 无最低置信度/箱体语义一致性门控，只返回数值；模型据此计算四个目标点。
- “destination wooden bin” 返回约 **768×779 mm** 的 XY 包络；多个别名定位近似同一小物体。不能把“API 有返回”当成目标箱识别成功。
- Bread 查询分数 **0.0177**，框与 Cereal 几乎重合；Bread 本体在该轮没有抬升。Cereal 自身的抬升只有约 11.7 mm。
- Milk 和 Can 的独立离线物体轨迹分别有 **143.9 mm、146.5 mm** 的抬升，说明原运动/抓取路径不是完全不能运行。但搬运后仍未进入目标区域，全部释放时原生计数仍为 0。
- 视频前、中、后帧见 `b0-video-contact-sheet.jpg`；实际物体留在原箱，右侧目标箱未完成。模型没有因 FAIL/FINISH 改变物理评分。

这支持“原始定位→目标计算→抓取/放置验证链路在此场景有缺口”，**不支持原始 CaP-X 普遍无能力，也不能证明新增保护全都正确**。没有调整 SAM3 阈值、模型输出或原始策略来筛选更好结果。

2 请求输入 3493、输出 15486 Token（含 reasoning 13129），配置费用上界 **US$0.06593532**，独立账本；不是服务商账单。

## 当前共同底座：LOGIV / G0 / G1 实际兼容性重放

`external-current-code-replay` 使用 V30c 的真实 DeepSeek 历史初始程序，不新增模型请求。三个方法共享完全相同初始程序、Snapshot、Vision-only、v30 Skill、控制器和预算。V30 Snapshot 与上述 B0/V36 Snapshot 的 model_hash、physics 和 state_hash 完全一致，spec 中策略配置不同。均在当前源码执行，运行中源码 hash 无变化。

| 方法 | 物理结果 | API / capture | SAM3 / GraspNet / IK | Skill / repair | 耗时 | 实际终止 |
|---|---|---:|---:|---:|---:|---|
| G0 | 1/4，完整失败 | 45 / 62 | 211 / 6 / 78 | 9 / 6 | 78.59 s | 阶段恢复预算耗尽 |
| G1 | 1/4，完整失败 | 38 / 49 | 171 / 4 / 50 | 6 / 3 | 61.52 s | 附着恢复中夹爪未收敛，无合法后续能力 |
| LOGIV 适配 | 1/4，完整失败 | 24 / 36 | 136 / 2 / 27 | 3 / 0（共同 runtime 计数） | 44.70 s | 官方核心 UNRESOLVED；第二对象 TCP 未收敛 |

LOGIV 实际派发 acquire → deposit → acquire，前两步完成，第三步失败，其后官方策略未找到可继续方案。没有偷偷调用 G0 的保守恢复代替官方决策。G0/G1 有实际撤退恢复成功事件，随后仍未完成第二个目标。G1 少 7 API、13 capture、40 SAM3，但 G0 尝试了更多恢复动作，不能仅以此证明选择性图推理效率更高；LOGIV 更早退出也不能称为更优。

本次没有把三组历史重放包装成新模型对照，也没有将三组结果与 B0 不同初始程序/恢复预算直接合并统计成功率。现阶段未有官方 ASPIRE/CaP-Agent0 同场景结果，外部主实验仍未齐备。

## V36 接口回归与检验

最新实际父进程探针另外发现并修复两项接线回归：

- 第一次 capture 之前 DSE 访问尚未存在的 `scene.obs`。现在返回明确未知测量，不为编码偷偷采集或伪造状态。
- 语义开关从冻结的历史环境 spec 读取，造成 Prompt 宣称有工具但实际 actor 没有注册。现在显式传入运行策略 policy，环境/Snapshot spec 保持不变。

连同语义日志 `kind` 参数冲突的修复，`harness-registration-native-policy` 在真实 Native 环境、父进程 Broker、嵌套 Bubblewrap 完成登记、两次实际公开查询与逐字段核验；`VERIFIED` 仅指忠实输出投影，`physical_verified=false`。该探针为手写零 LLM 接口验收，不计为模型物理恢复。

最终 **230 passed / 0 skipped / 0 failed**（相关 P0、V33–V36、统一入口、官方 LOGIV 核心桥接，非全量 Graph suite）。`delivery-baseline-tests.xml/log` 保存结果。中间一次测试与实际实验争抢原全局锁，出现 `Campaign already running`；保留失败日志，实验结束后单独补验和最终全套均通过，没有删除锁或弱化测试。`git diff --check` 通过。

## 复跑命令

真实 B0 最小适配（新目录，最多 2 次付费请求）：

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
export MPLCONFIGDIR=/tmp/capx-mpl NUMBA_CACHE_DIR=/tmp/capx-numba
scripts/local/infer.sh run --task pick_place --methods B0 --seeds 0 \
  --snapshot outputs/model_comparison/graph_v34_20261125_153341/open-g0-protocol-final/seed-0-repeat-0/G0/snapshot.json \
  --config env_configs/graph_feedback/development-v36.json \
  --max-requests 2 --max-blocks 2 --max-tokens 32768 --reasoning-effort low \
  --request-timeout 400 --wall-seconds 1500 --cost-limit-usd 1 \
  --output "outputs/inference/b0_minimal_$(date +%Y%m%d_%H%M%S)"
```

当前共同 Skill 的兼容性重放（零新 LLM）：

```bash
scripts/local/infer.sh replay --task pick_place --methods G0 G1 LOGIV --seeds 0 \
  --reference outputs/model_comparison/graph_v30_20261124_185019/online-v30c-normal0 \
  --snapshot-group G0 --skill v30 --harness legacy --force-mode off \
  --config outputs/model_comparison/graph_v36_20261125_175831/external-compat-config.json \
  --output "outputs/inference/external_replay_$(date +%Y%m%d_%H%M%S)"
```

服务与官方 LOGIV checkout 从原 `env_configs/inference/local-v31.json` 读取，可用原 `--ik-runtime/--external-logiv` 覆盖；未新增硬编码服务地址。B0 新任务路径目前要求单独执行和显式 Snapshot，以防误用其他方法的代码；Web 不宣称已做该路径交互验收。

**后续条件：** 先补原策略在合法程序执行后的多轮物理失败反馈、公开箱体实例与低置信度返回诊断；再以明确的同工具适配版本进行方法配对。官方原版、最小任务适配、共同底座适配必须各自记账和标注。完整 V36 物理底座与外部正式评测均仍为 **NO-GO**。

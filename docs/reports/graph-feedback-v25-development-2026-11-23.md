# Graph V25 开发与公开任务接入报告

日期：2026-11-23。正式工程 `/home/hanyu/dqb/cap-x-20261106`；分支 `feature/graph-complete-20261120`，HEAD `73be0eb6cb5e0756265254ad155f9d022e4e7e75`，未提交工作保留。本轮未提交或推送 Git。

**结论：公开任务下载和真实环境验收已完成，Vision Skill 有局部进展；V25 要求的完整 Vision 长程任务及真实 DeepSeek 消融尚未完成，扩大模型实验 NO-GO。** 本报告不将参考动作、Oracle、机制测试或局部成功标作模型任务成功。

全部本轮产物：`outputs/model_comparison/graph_v25_20261123_190518/`，以下称 R25。所有运行使用新子目录，早期失败、源码归档、图、物理 Trace、公开 RGB-D、视频及 HTTP 记录均保留。命令见 [V25 使用说明](../feedback/graph-v25.md)。

## 1. 按最新要求优先使用上游任务

没有自行创建新组合环境。下载并核验：

- [NVlabs/MimicGen](https://github.com/NVlabs/mimicgen)，提交 `c0fbff22ba40060f05a917a80817f3f5c20731e7`。
- [Robosuite](https://github.com/ARISE-Initiative/robosuite)，MimicGen 文档指定的 1.4.1 提交 `b9d8d3de5e3dfd1724f4a0e6555246c460407daa`。
- 两份上游人类参考 HDF5，共 108,834,245 bytes，数据版本 `33016f8a62c02334f929f2913af8fdd2a8a129e1`。每份 URL、大小与 SHA256 在 `env_configs/upstream_tasks/mimicgen-v25.json`。

下载器不安装依赖、不覆盖已有 checkout；发现版本、工作树或数据摘要不符就停止。MimicGen 的 NVIDIA License 限研究/评估用途，原许可证保留；不把它宣称为无限制商业开源许可证。主 `.venv` 的 Robosuite 1.5.1、MuJoCo 3.5.0 及模型依赖均未升级。

独立任务进程临时加载上游 1.4.1 源码，并复用项目已有 EGL 上下文实现以解决 ctypes 渲染兼容问题。未修改上游物理环境和 `_check_success()`。该适配是参考环境入口，**不等于已完成上游环境与 CaP-X Vision 控制边界的接入**。

| 上游任务 | 环境 Gate | 参考动作与原生结果 | 当前意义 |
|---|---|---|---|
| ThreePieceAssembly | 同 Seed 重建状态一致；初始失败；实际 step/渲染通过 | `demo_0`，329 次连续动作，两次回放均成功；复验 14.37 s | 可执行的连续精细装配任务候选 |
| CoffeePreparation | 同上 | `demo_0`，676 次动作，两次最终均失败；复验 28.33 s；杯子到位，插入未完成 | 真实多阶段任务，插入/关节操作仍需能力 |
| Threading | 同上 | 未运行完整示范或 Oracle | 精细操作候选，仅环境级验收 |

参考动作来自上游人类示范，**不是 Oracle、Vision Agent 或 DeepSeek 输出**。每次只在开始恢复一次数据集物理状态，随后逐动作 `env.step()`，中途状态重置为 0。两次相同示范不是两个独立场景。

证据：R25 `upstream-release-gate/`、`upstream-assembly-reference-{1,2}/`、`upstream-coffee-reference-{1,2}/`、`upstream-release-verification.log`。最后一组入口保存源码副本、固定 commit、数据摘要、原生评分、动作进度和视频。早期 Gate 的 `snapshot_replay` 字段只表示同 Seed 重建一致；最终入口已明确改名 `same_seed_initial_state`，不宣称已验证中途控制器续接。

CoffeePreparation 的上游阶段包括杯子搬运、抽屉打开、胶囊抓取和机器操作；ThreePieceAssembly 包括两件零件依次装配及保持前一装配效果。它们能提供真实环境约束，但尚未接入 Graph Skill，故不编造其图深度、分叉数或图收益。

## 2. 共同 Vision 能力的实际修改

继续使用已有父进程边界、真实 SAM3 / RGB-D / GraspNet / PyRoKi、技能契约和 Sandbox。未新增 Agent 框架，未修改 G1 的核心执行器，也未加入任务名、颜色、Seed 或故障标签分派规则。

新增/扩展：

1. 公开多视角实例绑定：处理嵌套重复掩码、不同实例歧义、别名有序回退、公开区域裁剪和小块深度断连污染。删除的实例拟合点不能用来证明自由空间。
2. 容器区域：使用当前深度测得的水平面与正交壁面；只对当前需要的容器角色补证，不能用历史边界填补缺失壁面。
3. Acquire / Deposit：有限抓取候选、现有坐标变换和 IK、TCP 实际收敛、夹爪编码器、当前几何持物验证、分段搬运、释放前完整区间检查、有限局部对齐和稳定放置验证。
4. 碰撞检查不再只选竖直壁面，还保留当前可见的水平容器边沿。该检查仍不能证明被遮挡体积或整个关节运动自由。
5. 非方块记忆使用明确标记的推断棱柱模型；当前 RGB-D 配准必不可少，预测只提出关联范围，不能伪装成观测。实际当前证据可以撤销已完成条件。
6. 场景内视觉目标标记与实物分开：释放后追踪同一实例的几何，不以同类静态标记的语义检测确认任务成功。
7. 已知技能边界停止与未知物理效果隔离分开；停止的 intent 不直接重放。未验证抓持不能被描述为机器人空闲。
8. `insertion_geometry.py` 可从公开点云提取孔、轴向和突出把手。它明确不提供插入或释放许可，尚无实际 NutAssembly Vision 动作闭环。

非方块误差包络来自 8 个初始开发观测：最大 XY 包络缺口约 5.75 mm，尺寸高度缺口约 1.11 mm；加离散化余量得到 XY 10 mm、Z 6 mm。状态明示 `limited_development_calibration_NOT_unseen_qualification`。这不是动态变形、所有姿态或未见对象的可靠性证明；不开放纯预测释放。原 Stack 释放阈值、Sandbox 和原生评分均不变。

## 3. 真实 Vision 参考执行结果

最终源码固定后，串行运行两个固定开发 Seed 的 G0/G1。每对同 Seed 从全新环境重建，相同初始物理状态 hash 和模型 hash；139 个归档文件运行后无变化。结果是**零新增 LLM 的参考技能 Gate**，不是共享 DeepSeek 初始程序的模型配对。

| Seed | 系统 | 原生完整成功 | 原生完成目标 | 边界 API / capture | 实际 HTTP | 端到端时间 |
|---|---|---|---|---|---|---|
| 0 | G0 | 否 | 1/4 | 26 / 35 | 225 | 56.87 s |
| 0 | G1 | 否 | 1/4 | 26 / 35 | 225 | 57.11 s |
| 21 | G0 | 否 | 0/4 | 0 / 1 | 2 | 2.02 s |
| 21 | G1 | 否 | 0/4 | 0 / 1 | 2 | 1.95 s |

Seed 0 每组 26 次边界调用包含 5 次 `execute_skill` 和 21 次内部原语：10 goto、7 wait、3 open、1 close，不能将它们全部叫作高层 Skill。完成 acquire/deposit 两个 intent，首个物体在当前视觉验证和独立原生评价中均到位。HTTP 为 SAM3 196、GraspNet 2、IK 27；SAM3 请求累计耗时 G0 28.28 s / G1 28.58 s。嵌套调用耗时不能相加当作总墙钟时间。

两对的图决策差异均为 **0**。同等子目标进展下调用数也相同。当前结果不支持图带来效率或成功率收益。当前参考 Gate 仅切换补证范围，并非完整强 G0 / G1 模型恢复消融；不能从两个受共同技能瓶颈限制的场景推断图是否有效。

证据：R25 `final-reference-{G0,G1}-seed-{0,21}/`、`final-reference-analysis.json`。四个 episode 完整失败全部保留，reward 为 0；未计作成功或排除分母。未使用 heldout Seeds。

## 4. 关键失败与修复证据

- **早期 TCP 残差：** 旧底层关节收敛阈值返回时，TCP 仍可能差约 7 mm。新技能在相同目标下有限等待并复测，保持 4 mm / 3° 检查，不靠降低阈值通过。
- **夹爪符号：** 两指编码器带相反符号，直接求和会将持物宽度误算为 0。改成经范围验证的两指绝对位移和，并增加反例测试。
- **释放区间：** `skill-attempt-12` 的第一个物体经过一次约 11–12 mm 的 XY、6.59 mm 的 Z 局部调整，间隙区间从不合法变为约 [2.15, 23.02] mm，然后才释放；原生该目标为成功。`skill-attempt-13` 补齐了释放后实例稳定验证。这些是共同技能的无 LLM 局部修复证据，不是图选择性恢复或模型恢复成功。
- **容器边沿碰撞：** `skill-attempt-14` Bread 下降时，离线记录出现 hand 与未命名环境 geom 的接触，TCP 误差约 11.7 mm。只保留竖直面会遗漏水平边沿，新增测试先红后绿后扩展了可见障碍面。`skill-attempt-15` 曾跨过该下降，但最终参考配对仍出现掌部接触/不收敛；公开掩码覆盖和完整扫掠可见性仍不足，不能宣称碰撞问题彻底解决。
- **实际附着变化：** `skill-attempt-15` Bread 的当前视觉配准 residual95 约 2.89 mm、coverage 约 99.84%，但旋转创新约 17.32°，超过原 8° 限制。离线独立物理诊断发现：抬升过程物体转动 18.29°、TCP 仅 0.19°、物体相对 TCP 变化约 18.19°。这支持阻断，不能仅因表面 residual 小就认为附着稳定。真值仅写入离线诊断文件，未反馈给在线技能。
- **Seed 21：** 初始公开壁面不足以确定容器边界，未派发机器人动作。未更换 Seed、填充真值或放宽识别阈值。

离线证据：R25 `bread-offline-contact-diagnostic.json` 和 `bread-offline-relative-rotation.json`。15 个开发中间运行是逐版诊断，不能拼成成功率或当作独立样本。最终参考配对与中间版本单独记录。

## 5. 测试与工程验收

最终：**1151 passed / 0 failed / 0 skipped**，包括 Graph 1083、P0 68；共 264.69 s。启用了 6 个真实历史 Snapshot 渲染测试与 P0 的两项真实视觉集成。新增 V25 测试 31 项，覆盖区域/实例、真实证据生命周期、夹爪/配准、区间许可、孔特征、下载完整性和测试环境隔离。

证据：R25 `regression-final.xml`、`regression-final.log`、`depth-runtime-final.json`、`regression-final-artifacts/`。单项测试与故意失败的修复前日志另行保留。单元测试通过不等于 Vision 任务通过。

首次合并回归为 1141 passed / 3 failed / 6 skipped，未删除：

1. 下载器配置不属于 Graph campaign，却放进了其配置扫描目录。已移动至 `env_configs/upstream_tasks/`；没有削弱源码冻结断言。
2. 既有 Mock campaign fixture 将 EGL 等变量永久留为 `fixture`，导致两项后续真实渲染测试失败。新增环境恢复测试复现，再用 fixture 内的环境上下文恢复解决；不修改业务渲染代码、不降低断言。
3. 最终提供实际历史 config / Snapshot，补齐 6 个渲染用例。

服务沿用配置中的 SAM3 18114、seeded GraspNet 8215、PyRoKi 41337，没有重启其他项目服务。NVML 仍报告驱动/库不匹配，无法可靠报告 GPU 显存；仿真使用现有软件 EGL，视觉服务实际响应已记录。未升级 CUDA、PyTorch、Transformers、vLLM、VeRL，未下载模型、训练或操作真实机器人。

## 6. 修改文件及保留工作

原有未提交文件在 R25 `source-before/`、`source-before.json` 归档，无原有文件被删除。相对此轮开始，修改的既有实现为：

- `capx/execution/skill_contracts.py`：已知边界停止、禁止旧 intent 重放。
- `capx/experiments/native_tasks.py`：父进程可信原语与有界等待，Vision actor 不开放原始运动权限。
- `capx/experiments/native_vision.py`：公开实例/区域观测与关联。
- `scripts/local/native_task_gate.py`：增加 Vision Skill 开发 Gate。
- `tests/graph_v2/test_campaign.py`：仅修复 Mock fixture 环境污染。
- `README.md`：增加当前入口和真实验收状态。

新增：`region_geometry.py`、`vision_skills.py`、`insertion_geometry.py`、`native_vision_skills.py`、两个上游下载/验收脚本、V25 开发/误差/上游配置、五个 `test_v25_*.py` 文件、本说明和使用文档。既有 `grounded_feedback.py`、`grounded_recovery.py`、`graph_executive.py` 等此前未提交改动保留，不能将其整份 Git diff 计作本轮新增。

## 7. V25 尚未完成的部分及准入判断

- 新任务完整 Vision 成功：**未获得**。PickPlace 仅完成一个目标；NutAssembly 仍止于特征能力。
- 连续复杂任务：ThreePieceAssembly **上游人类参考回放成功**；Graph/Vision/DeepSeek 成功均未验证。CoffeePreparation 参考回放仍失败。
- 新任务完整模型执行/恢复接口、B0、G1-no-selective、G1-no-future：**未完成**。现有原图实验框架和 Provider 保留，没有伪造组别或补写结果。
- 本轮新增 DeepSeek 请求、Token、付费 API 费用均为 **0**。原因是约定的物理 Gate 未通过，不是缺少密钥或忽略模型实验要求。
- 无独立证据归因 Future Intent、选择性恢复或状态记忆的增量收益。公开 RGB-D 与形状记忆支持局部动作，但同时暴露真实滑移、遮挡和可见扫掠局限。
- 不扩大固定开发集或正式未见测试：**NO-GO**。下一步应优先在下载任务中复用可验证控制与视觉技能，并补齐稳定抓持、带完整几何证据的可达性、非方块误差标定和精细插入；不能通过增加任务专用恢复分支绕过这些缺口。

本轮提供的是可复跑的阶段开发交付，不是已完成全部 V25 研究目标的声明。

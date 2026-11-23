# V25：公开任务复用与 Vision 技能开发验收

当前版本**尚未通过新的长程任务 Vision Gate，不可当作 DeepSeek 消融完成版**。入口复用 V24 的 `ContractGraph → GraphExecutive → ContractBroker → SandboxProcess`；原 Stack 路径、物理评分和历史结果保留。

按本轮新增要求，优先下载并复用上游任务，不自行搭建组合场景。

## 公开任务

| 来源 / 任务 | 当前实际验证 | 尚缺能力 |
|---|---|---|
| Robosuite PickPlace | 原生 Oracle；真实 Vision 完成一个物体的搬运、释放及稳定验证 | 四物体完整成功、可靠非方块附着模型、容器遮挡下的证据 |
| Robosuite NutAssemblySquare | 历史 Oracle；当前公开点云的孔/把手特征测试 | 经误差标定的插入技能、真实 Vision 完整成功 |
| MimicGen ThreePieceAssembly | 环境重建；上游人类示范 329 次连续动作，原生物理成功 | Vision Skill、Graph/DeepSeek 接入 |
| MimicGen CoffeePreparation | 环境重建；上游示范 676 次动作，杯子到位、最终失败 | 插入与关节操作能力、Vision/Graph 接入 |
| MimicGen Threading | 环境重建与实际 step | 参考物理成功、Vision 与精细控制 |

MimicGen 源码及原生任务由 [NVlabs/MimicGen](https://github.com/NVlabs/mimicgen) 提供。其 NVIDIA License 限研究/评估用途，并非不受限制的商业开源许可证；源码中的 LICENSE 原样保留。依照其安装说明，使用固定 Robosuite 1.4.1 源码。该目录只在独立任务进程的 `sys.path` 中启用，主 `.venv` 的 Robosuite 1.5.1 未替换。

`env_configs/upstream_tasks/mimicgen-v25.json` 固定上游提交、参考数据 URL/SHA256、渲染适配及运行设置。示范是数据，不是模型权重。没有安装依赖、下载模型或启动训练。

## 可执行命令

以下入口均不调用 LLM。输出目录必须是新目录，已存在时拒绝覆盖。脚本退出正常只表示完成了诊断，应读取 JSON 的 `passed` / `raw_success` 判断任务结果。

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
export PYTHONDONTWRITEBYTECODE=1
RUN="$PWD/outputs/model_comparison/v25_$(date +%Y%m%d_%H%M%S)"

# 下载固定版本的上游任务及两份参考数据；已有文件只核验，不覆盖。
.venv/bin/python -B scripts/local/fetch_upstream_tasks.py \
  --config env_configs/upstream_tasks/mimicgen-v25.json --with-reference-data

# 三个上游环境：同 Seed 重建、初始物理评分、实际 step、首帧。
.venv/bin/python -B scripts/local/external_task_gate.py \
  --config env_configs/upstream_tasks/mimicgen-v25.json \
  --tasks CoffeePreparation ThreePieceAssembly Threading --seed 0 \
  --output "$RUN/upstream-environments"

# 连续物理回放：只有一次初始状态恢复，不逐帧设置真值。
# 明确是上游人类动作参考，不能叫 Vision Agent 或 DeepSeek 实验。
.venv/bin/python -B scripts/local/external_task_gate.py \
  --config env_configs/upstream_tasks/mimicgen-v25.json \
  --demo capx/third_party/mimicgen_reference_data/three_piece_assembly.hdf5 \
  --episode demo_0 --output "$RUN/assembly-reference"

.venv/bin/python -B scripts/local/external_task_gate.py \
  --config env_configs/upstream_tasks/mimicgen-v25.json \
  --demo capx/third_party/mimicgen_reference_data/coffee_preparation.hdf5 \
  --episode demo_0 --output "$RUN/coffee-reference"

# 主环境中的实际 SAM3 / RGB-D / GraspNet / PyRoKi 技能 Gate。
# 目前 vision-skills 仅实现 pick_place，其他任务不要套此入口。
for GROUP in G0 G1; do
  .venv/bin/python -B scripts/local/native_task_gate.py \
    --config env_configs/graph_feedback/development-v25.json \
    --tasks pick_place --gate vision-skills --group "$GROUP" --seed 0 \
    --output "$RUN/pickplace-$GROUP"
done

# 新增机制与下载完整性测试，不需要在线服务。
.venv/bin/python -B -m pytest -q tests/graph_v2/test_v25_*.py
```

服务地址读取开发配置，未重新启动 SAM3、GraspNet 或 PyRoKi。配置沿用当前可运行的软件 EGL 方案。`--seed 21` 是固定的第二开发场景；heldout 场景未使用。

## 能力及证据边界

- `PublicScene` 使用公开相机图像、深度、标定、机器人反馈；候选歧义时拒绝选择。源容器和目标容器分别补证，避免无关区域遮挡阻止已许可的目标动作。
- `RegionSkills` 是共同的开发中能力。先检查可达性和公开障碍物，再执行分阶段 acquire/deposit；几何抓取候选与 GraspNet 候选分别记录来源。
- 历史几何只生成关联假设。当前 RGB-D 必须支持刚体配准，夹爪编码器和几何一致性共同验证持物。非方块误差配置只经过少量开发观测，**不是未见场景资格证书**，不开放纯预测释放。
- 释放使用最后一次采集后的持物与目标区间；仅在完整区间满足原技能契约时派发。局部调整有距离、次数、进展与可达性上限。
- 容器碰撞检查包含当前可见水平边沿。没有看到的体积仍不等于无障碍；当前检查不能证明整条关节运动无碰撞。
- 已完成放置通过保留的实例模型与当前深度重新核对，避免把原生场景中的视觉目标标记当成已放置实物。缺少当前证据时回到 UNKNOWN。
- 原生评分只由可信父进程读取，不能通过 `FINISH`、contract 完成或技能返回冒充物理成功。
- 当前参考 Gate 的 G0/G1 只有补证范围策略差别；尚未实现新任务的完整模型恢复、B0 和两项模型消融。因此这组 Gate 不能作为图收益的正式对照。

完整记录见 [V25 开发报告](../reports/graph-feedback-v25-development-2026-11-23.md)。新任务的 DeepSeek 付费准入仍关闭；现有 Provider、API Key、预算及原图实验入口保持原状。

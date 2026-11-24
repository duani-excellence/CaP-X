# V28 感知归因与代价感知补证

继续使用正式工程 `.venv`、现有 API 密钥及真实 SAM3/GraspNet。此入口会实际调用 DeepSeek，并产生费用；不使用 Oracle 初始程序。输出必须是新目录。

## 真实小规模配对

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN="$PWD/outputs/model_comparison/v28_$(date +%Y%m%d_%H%M%S)"
IK_RUNTIME="$PWD/outputs/model_comparison/graph_v28_20261124_155408/ik-runtime.json"
.venv/bin/python -B -m capx.experiments.contract_models \
  --config env_configs/graph_feedback/development-v28.json \
  --ik-runtime "$IK_RUNTIME" --seed 0 --study normal \
  --ablations no_graph_repair no_selective no_cost --perception-controls \
  --output "$RUN/normal-seed0"
```

`IK_RUNTIME` 是已运行服务生成的文件，不保证机器重启后仍在线。`--ik-runtime` 覆盖配置中的历史 PyRoKi 地址，同时归档实际配置。SAM3/GraspNet 地址来自配置；服务参数必须按当前环境检查。需要启动独立 CPU IK 时，复用现有入口，另开终端：

```bash
source scripts/local/common.sh
.venv/bin/python -B scripts/local/start_feedback_ik.py \
  --runtime-file "$PWD/outputs/model_comparison/ik_$(date +%Y%m%d_%H%M%S).json" \
  --warmup-requests "$PWD/outputs/model_comparison/graph_v27_20261123_235033/p0-acceptance-artifacts/replay/841c3fa264fe4528b81dc4b958cbeb51/requests.jsonl"
```

将上一步新文件传给 `--ik-runtime`。该脚本使用原 IK 求解器、CPU、操作系统分配的空闲端口，不停止其他服务。

上面的单条实验命令运行 7 个 episode：G0、G1、no_graph_repair、no_selective、no_cost，以及关闭共同感知优化和 V28 证据规划的 V27-G0/G1 策略对照。移除 `no_cost --perception-controls` 后是主要 4 组。固定开发场景为 Seed 0、21、7；来源过期压力测试用 `--study evidence_expiry`，与 normal 分开。各组共享一次新生成的 DeepSeek 初始程序和严格相同的完整 Snapshot；恢复代码依实际需要再次请求模型。

默认每个实验单元上限 24 次模型请求、配置估算 US$5、7200 秒；每 episode 6 个模型块、24 个 Skill、256 次 API、1200 秒、12 次补证。输出 Token 上限 32768、reasoning effort=low。所有退出和失败保留，原生物理成功判据不变。该费用是项目估价，不是服务商账单。

## 消融含义

|组|共同精确复用|证据范围与代价|恢复授权|
|---|---|---|---|
|G0|开启|强保守未完成后缀|模型可选所有合法能力|
|G1|开启|条件回归 + 共享采集需求 + 生命周期代价|图限定合法恢复范围|
|no_graph_repair|开启|与 G1 相同|关闭图范围限制，底层安全不变|
|no_selective|开启|保守后缀补证|保留图授权|
|no_cost|开启|V27 条件回归，不做代价合并|保留图授权|
|V27-G0/G1|关闭|V27 对应策略|V27 对应策略|

V27 对照使用同一当前可执行文件中的版本开关，保留新审计字段；并非另行检出旧仓库。对照的仿真参数、预算、任务定义和 Prompt 必须完全相同，只有执行策略元数据可不同。底层仍核对完整 Snapshot，不能忽略状态差异。

## 安全与复用边界

- 语义缓存仅复用完全相同 RGB、查询、相机和标定；缓存空结果但不缓存异常。当前深度仍重新投影，不能用旧深度替代。
- 每次验证仍获取真实当前 RGB-D/机器人状态；不跳过最终抓持、放置或释放检查。
- 同一帧的点云/几何可共享。跨帧的抓取计划还必须具有相同完整公开观测、运动版本、阶段及抓持/放置效果；任何变化重新计算。
- G1 的补证规划合并当前非瞬态证据需求，并只免费加入共享相同工具特征的未来条件。若已有一个完整合法分支就绪，优先将其交给恢复执行，不为无关目标继续补齐会随运动失效的 grasp/IK 计划。TRUE、已耗尽或无合法恢复路线的条件不反复拍摄。
- 代价是共同技能注册表的可解释估计，不是学习模型；真实耗时在工具日志单独记录。新增来源 ID 或缓存命中不等于信息增益，条件状态/实际可执行性变化需单独统计。
- 多别名、Crop、双视角和当前深度刚体一致性检验仍保留。第一版没有证明可以安全省掉这些不同输入的语义请求。

## 只读分析与测试

```bash
.venv/bin/python -B scripts/local/analyze_perception_cost.py \
  "$RUN/normal-seed0" --output "$RUN/perception.json"
.venv/bin/python -B scripts/local/analyze_contract_models.py \
  "$RUN/normal-seed0" --output "$RUN/contracts.json"
.venv/bin/python -B -m pytest -q tests/graph_v2/test_v28_perception_cost.py
```

逐次请求分析会核对工具 NPZ 的输入图像与归档 RGB/Crop 字节相同，拒绝错位归因。旧日志的阶段由邻接 Trace 推断，V28 使用真实包装器调用栈。`perception_request` 记录上下文、输入键、当前来源、缓存原始来源和实际/复用耗时；`perception_plan_reuse` 记录实际计划复用；`perception_verification` 记录条件前后状态；`perception_frame` 记录 capture+归档耗时（不是纯相机计算耗时）。GraspNet/IK 实际 HTTP 数量和时间见第二个分析文件。

开发结果及局限见 [V28 报告](../reports/graph-feedback-v28-development-2026-11-24.md)。

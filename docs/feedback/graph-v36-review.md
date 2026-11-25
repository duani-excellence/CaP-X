# V36 独立审查后的修复与验收

正式目录 `/home/hanyu/dqb/cap-x-20261106`；沿用当前分支。复用原统一入口、原 Snapshot、Sandbox、Broker、独立原生评分。配置：`env_configs/graph_feedback/development-v36-review.json`。这仍是开发版，不是完成四物体任务或新轨迹后端验收的声明。

## 新 DeepSeek 配对（会产生费用）

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/capx-review-mpl NUMBA_CACHE_DIR=/tmp/capx-review-numba
RUN="$PWD/outputs/inference/v36_review_$(date +%Y%m%d_%H%M%S)"
scripts/local/infer.sh run \
  --task pick_place --methods G0 G1 --seeds 0 \
  --harness open --skill v32 --force-mode on \
  --config env_configs/graph_feedback/development-v36-review.json \
  --snapshot outputs/model_comparison/graph_v34_20261125_153341/open-g0-protocol-final/seed-0-repeat-0/G0/snapshot.json \
  --decision-encoding encoded --recovery-requests 3 \
  --max-requests 7 --cost-limit-usd 3 --wall-seconds 3000 \
  --episode-seconds 1200 --max-blocks 12 --output "$RUN"
```

两组共享一次初始生成和同一物理 Snapshot，恢复模型请求各受上限约束。7 次是整个运行的请求总额，初始请求不会在第二组重复付费。不要修改运行中的源码。后续配置修改请使用新输出目录；保留失败及账本。实际服务位置由现有 `env_configs/inference/local-v31.json` 和配置加载，不在算法中固定端口。

## 零新增 LLM 的真实 Vision 重放

```bash
scripts/local/infer.sh run \
  --task pick_place --methods G0 G1 --seeds 0 \
  --harness open --skill v32 --force-mode on \
  --config env_configs/graph_feedback/development-v36-review.json \
  --reference outputs/model_comparison/graph_v36_20261125_175831/final-source-replay \
  --snapshot-group G0 --recovery-requests 0 \
  --max-blocks 12 --episode-seconds 1200 \
  --output "$PWD/outputs/inference/v36_review_replay_$(date +%Y%m%d_%H%M%S)"
```

重放不会请求新模型；旧 `llm_requests` 字段可能包含归档初始请求引用，不能据此计为新增付费请求。以本次账本、请求原始回执与程序 origin 为准。

## 软件回归

```bash
export CAPX_LOGIV_SOURCE="$PWD/outputs/model_comparison/graph_v30_20261124_185019/upstream/LOGIV-official"
.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/graph_v2/test_v36_review_repair.py \
  tests/graph_v2/test_v36_semantic.py \
  tests/graph_v2/test_v29_causal_repair.py \
  tests/graph_v2/test_v30_external_bridge.py
```

新增回归来自审查反例；原附件探针原样保存在新实验目录，原探针的缺陷断言在修复后失败是正常的，不能把它们当作通过标准。

## 语义及限制

- 当前恢复进展在必要补证结束后计算，每轮一次。重新建立旧条件/负前置条件可以形成进展，重复状态、仅换来源 ID、无目的位移不能无限重置终止计数。
- 父进程可信验证失败传播明确的条件 effects，保留可能持有的对象身份；参数错误不等于物理失效。
- Native 控制步记录真实派发计数。规划前拒绝为 NOT_DISPATCHED；已派发且能核对现场的受控停止为 DISPATCHED_RECONCILED；无法核对仍隔离。
- `plan_skill` 产生条件性的未来候选；执行选择才提交到活动计划。无条件字面量调用可提前编译；任意 Python 分支保持开放，不伪造总执行顺序。
- 动态能力可调用与可规划分开。只对最多四个静态绑定、已有可信技能的直线组合推导组合合同；复杂分支、动态绑定保持 CALLABLE_ONLY。注册不证明物理成功。
- 摘要输出通过只证明其忠实引用，最高保持限定范围 VERIFIED。不同记录 ID、两次不同读数或两个 episode 都不足以自动宣称 REUSABLE。
- `preview_skill('acquire', ...)` 的 `selected.candidate_id` 才是可选择的抓取候选。普通 `observe_grasp().id` 是观测结果 ID，不能当候选 ID。执行时重新检查同一候选；变更则拒绝。联合预览默认最多三种抓法、每抓法八个放置候选；可用 `future_positions=1..12` 指定每个直立 yaw 的位置数（4 个 yaw），最大每抓法 48 个。`planning_query(id).joint_search` 保留失败残差和已测试候选。有限未找到解不证明物理不可达。
- 原 GraspNet 完整 6D 姿态仍为诊断信息，未绕过 TCP 标定及接近验证。
- 新 PyRoKi 轨迹后端已接入当前公开障碍；持物与机器人碰撞模型及完整自由空间仍有缺口，不标为成熟 Skill 的可用替代品。不会通过放宽阈值解决 UNKNOWN。
- 每组动态能力库默认位于本 episode 内。受控故障通过统一技能完成钩子触发，未触发必须标为 scenario_not_activated。

详见 [开发报告](../reports/graph-feedback-v36-reviewed-repair-2026-11-25.md)。


## 重放已生成恢复代码；默认不再调用模型

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
REF=outputs/model_comparison/v36_review_repair_20261125_200443/live-pair/seed-0-repeat-0
.venv/bin/python -B scripts/local/harness_probe.py \
  --reference "$REF" --snapshot-group G1 --group G1 --harness open \
  --ik-runtime outputs/model_comparison/graph_v33_20261125_000920/ik-runtime.json \
  --config env_configs/graph_feedback/development-v36-review.json \
  --decision-encoding encoded \
  --recorded-recovery "$REF/G1/actor-03-program.json" \
  --recorded-recovery "$REF/G1/actor-06-program.json" \
  --output "$PWD/outputs/inference/review_recorded_$(date +%Y%m%d_%H%M%S)"
```

上述调用为零新增 LLM。只有显式附加 `--live-recovery-requests 1` 才允许在归档前缀后新增一次付费恢复请求；这是混合来源诊断，不能记成一次全新的模型初始生成。全部程序仍由原 Sandbox/Broker/Runner 执行，恢复不回滚物理状态，不重新请求已有程序。实际服务运行文件来自当前本地配置；更换机器时先核对该文件，不能把示例端口当固定算法参数。

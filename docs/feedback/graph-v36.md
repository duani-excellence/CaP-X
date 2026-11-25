# V36：运行方式与验收边界

继续使用正式工程和当前分支；所有输出路径必须是新目录。V36 没有修改原生评分，没有新增仿真器，没有升级主 `.venv`。

## 真实模型 + Vision + F/T

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/capx-mpl NUMBA_CACHE_DIR=/tmp/capx-numba
RUN="$PWD/outputs/inference/v36_$(date +%Y%m%d_%H%M%S)"
scripts/local/infer.sh run \
  --task pick_place --methods G0 --seeds 0 --harness open \
  --skill v32 --force-mode on \
  --config env_configs/graph_feedback/development-v36.json \
  --snapshot outputs/model_comparison/graph_v34_20261125_153341/open-g0-protocol-final/seed-0-repeat-0/G0/snapshot.json \
  --decision-encoding encoded --recovery-requests 3 --max-requests 4 \
  --cost-limit-usd 2.0 --wall-seconds 1800 --episode-seconds 1200 \
  --max-blocks 12 --output "$RUN"
```

这是付费真实请求。当前默认仍是保留的成熟 Skill 运动路径；**不能因此宣称新的全轨迹底座已通过**。`--methods G0 G1` 仍由原 Runner 共享 Snapshot 和初始程序，预算是整个运行的总额，并非每个方法各自的额外额度。本轮只做小规模验收，未扩大正式评测。

## 冻结公开状态上的真实模型对照

Gate B/C 无仿真动作，使用归档公开测量、真实 DeepSeek 和现有在线 IK；不是物理恢复成功实验。

```bash
OUT="$PWD/outputs/model_comparison/v36_decision_$(date +%Y%m%d_%H%M%S)"
for MODE in legacy encoded semantic; do
  .venv/bin/python -B scripts/local/decision_semantic_gate.py \
    --diagnosis outputs/model_comparison/graph_v35_20261125_170349/can-zero-llm/diagnosis.json \
    --target 3 \
    --ik-runtime outputs/model_comparison/graph_v33_20261125_000920/ik-runtime.json \
    --config env_configs/graph_feedback/development-v36.json \
    --ledger "$OUT/ledger" --output "$OUT/$MODE" --mode "$MODE" --rounds 2
done
```

上述路径是实际已归档的开发案例，不是核心代码里的固定路径。新案例通过参数传入。共享账本防止重发同 label/prompt；不确定的已发送请求需要核对 receipt，不自动重试。原 messages 在 `requests/*/llm/*/prompt.json`，响应、HTTP 状态和 usage 同目录保存。

## 动态能力与工具契约

实际接口定义由 `open_recovery.runtime_api_doc()` 统一提供给初始与恢复 Prompt。`capability_tools()` 返回真正绑定的 API、签名、版本和物理副作用，而非 LLM 猜测的工具目录。

- `register_capability(proposal)`：校验 Schema、函数签名、实际调用名、副作用及验证引用。错误返回 REJECTED，可修订。不会写入可信条件或授予许可。
- `run_capability(id, params, evidence_ids=[])`：在现有 Bubblewrap 中运行组合函数，每次工具调用仍通过原 Broker、预算和安全检查。
- `capability_query(id=None)`：检索持久化版本及运行记录。实现版本变化时 DEPRECATED；旧证据不能作为新调用的有效输入。
- DATA 依赖由实际 RPC/显式证据 ID 建立；语义关系标为 `SEMANTIC_DEPENDENCY` 假设。没有干预证据时不生成物理因果结论。

注册成功不等于执行成功，输出类型正确不等于摘要正确，执行成功不等于物理效果或任务成功。查看 `verification_scope`、`physical_verified` 和独立评分，不能仅看节点数量或状态名称。

最终版中，返回类型合格只到 `EXECUTED`。`output_projections` 允许父进程对照实际 RPC 输出核验摘要字段（包括 `summary.status` 形式的嵌套字段）；必须覆盖全部输出，才得到限定为 `audited_output_projections` 的 `VERIFIED`。它不证明摘要策略有效。跨目录重复同一冻结状态不能升级为 `REUSABLE`；还需不同实际工具结果与 episode，且验证范围仍不变。物理效果只能由原父进程条件验证器确认。

`decision_state().attachment` 与 `execution_state().attachment` 均为对象名称字符串或 `None`。详细附着估计单独放在 `attachment_estimate`，不可对字符串调用 `.get()`。最终真实模型运行曾遇到这个类型混淆，修复后的契约有回归测试；没有把修复前的失败重写为成功。

零付费重放一份模型生成的组合能力：在前述 `decision_semantic_gate.py` 命令中使用 `--mode semantic --rounds 1 --replay-program <program-0.json>`，输出必须换成新目录。结果标为 `frozen_public_archived_model_replay`，不会调用 LLM，不计为新模型实验。

## 可选成熟轨迹底座

在新的配置副本中设置 `tasks.pick_place.motion_backend = "pyroki_public"`，成熟 Skill 与开放 `plan_motion/run_plan` 均接到共享适配器。后者真正预生成轨迹，执行阶段消费该轨迹，检查起点版本并逐个派发 waypoint。没有通过公开扫掠验证时不派发。

使用已有 PyRoKi CPU TrajOpt，不把 `/plan` 的线性 IK 插值当作碰撞规划。球包络的重叠可用 FCL 对配置碰撞网格做窄相检测；FCL 不可用时保持 UNKNOWN。FCL 是独立可选依赖，例如：

```bash
DEPS="$PWD/outputs/optional_deps/fcl_$(date +%Y%m%d_%H%M%S)"
UV_CACHE_DIR=/tmp/capx-uv uv pip install --python .venv/bin/python \
  --target "$DEPS" --no-deps --only-binary :all: python-fcl==0.7.0.10
export PYTHONPATH="$DEPS:$PYTHONPATH"
```

本轮实际发现并修正了固定组件碰撞对和非控制夹爪关节的起点绑定问题。但是公开环境扫掠、持物与机器人碰撞合同仍不完整，**共享后端尚无获准派发的真实轨迹正例**。不能用 `PASS` 的自碰撞结果代替环境/持物验证；不能因旧 Skill 可执行就声称新底座已获验收。

上游依据：[PyRoKi](https://github.com/chungmin99/pyroki)、[python-fcl](https://github.com/BerkeleyAutomation/python-fcl)。详细结果见 [V36 报告](../reports/graph-feedback-v36-development-2026-11-25.md)。


## Baseline 与最小原始策略入口

B0 PickPlace 已接到同一 `scripts/local/infer.sh`，走原程序级 Runner；要求单独 `--methods B0 --snapshot <精确快照>`，`--skill auto/raw`，使用自己的原始 API 程序。它是 P0 兼容的最小任务适配，不是 CaP-Agent0 或官方逐字复现。G0/G1 仍保留各自策略；LOGIV 保留官方执行/修复核心，只支持其已有 legacy Skill 适配。ASPIRE/CaP-Agent0 尚不可用，CLI 不会伪装支持。

真实可运行命令、当前能力矩阵和同场景诊断见 [Baseline 报告](../reports/v36-baseline-compatibility-2026-11-25.md)。B0 需要付费生成，历史 LOGIV/G0/G1 兼容性重放零新 LLM。不要把 open-runtime 配置直接用作 LOGIV/legacy；入口会拒绝而不是静默忽略语义工具。当前环境策略开关从运行 policy 读取，不改冻结 Snapshot spec。

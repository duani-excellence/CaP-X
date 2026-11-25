# V34 Open Agent Runtime

在正式工程根目录运行。沿用 V31 统一 Runner、V33 Harness、Sandbox、Broker、Vision、F/T、Snapshot 和独立原生评分。`--harness open` 启用本轮能力；`persistent`、`legacy` 保留旧路径。本版本仍是开发原型，不能当作通用碰撞规划或真机安全验证。

## 新 DeepSeek 初始程序 + 多轮恢复

```bash
cd /home/hanyu/dqb/cap-x-20261106
scripts/local/infer.sh run \
  --methods G0 --harness open --skill v32 --force-mode on --seeds 0 \
  --snapshot outputs/model_comparison/graph_v32_20261124_221234/online-fourway-final/seed-0-repeat-0/D-FT-G1/snapshot.json \
  --recovery-requests 6 --max-requests 7 \
  --output "outputs/model_comparison/v34_open_$(date +%Y%m%d_%H%M%S)"
```

命令会产生 DeepSeek 费用。默认开发配置 `env_configs/graph_feedback/development-v34.json`：32768 输出 token、low reasoning、最多 7 请求、2 USD 估算费用上界、2400 秒活动预算。每 episode 1200 秒、256 个机器人 API、24 次 Skill、最多 18 程序块。`--recovery-requests` 是 LLM 恢复请求上限，不是成功次数。确定性恢复同样占用物理预算。请求无无限自动重试。可通过统一 `run --help` 查看覆盖参数。

服务和 IK 读取现有配置与 `env_configs/inference/local-v31.json`，不要另起重复 GPU 模型。已有服务不可用时明确报错。`--methods G0 G1` 可做配对；本轮先验收 G0，没有宣称 G1 优势。

## 历史程序 + 同一 Snapshot，零新 LLM

```bash
scripts/local/infer.sh replay \
  --methods G0 --harness open --skill v32 --force-mode on \
  --reference outputs/model_comparison/graph_v32_20261124_221234/online-fourway-final/seed-0-repeat-0 \
  --snapshot-group D-FT-G1 --recovery-requests 0 \
  --output "outputs/model_comparison/v34_replay_$(date +%Y%m%d_%H%M%S)"
```

`--reference` 配合 `--recovery-requests 0` 使用 NoNewLLM；不要传 `--max-requests 0`（该参数要求至少 1）。历史初始 receipt 可能仍出现在结果中；新增调用数看 `new_llm_requests` 与本轮 request ledger。

## Coding Agent API

初始与恢复程序共用 `open_recovery.OPEN_API_DOC`。允许函数、变量、计算、条件、有界循环；不按固定 Skill ID 排列程序。参数化 `plan_skill` 经父进程注册契约；`run_plan` 才检查与执行。

```python
p = plan_skill('acquire', params={'object': object_name})
if p.get('status') == 'READY_FOR_LIVE_VERIFICATION':
    run_plan(p['id'])
```

`READY_FOR_LIVE_VERIFICATION` 不是执行许可；不应判断为 `ok`。`ALREADY_SATISFIED` / `REJECTED` 没有可执行计划 ID。成熟 Skill 自行收集必要证据，不必每次重新查询全部对象与容器。显式 `inputs` 只接受本公开 API 返回且仍有效的 ID。

- 观测：`read_robot`、`observe_object`、`observe_grasp`、`observe_surfaces`。
- 记忆：`belief_query`、`history_query`、`recall_result`、既有 `graph_query`。
- 动作：`plan_skill/run_plan`、`plan_motion/run_plan`、`move_tcp`。
- 信息动作建议：`observation_opportunities(role)`。基于已有壁面射线、固定相机和当前 TCP 的遮挡假设，只提出候选，不能派发运动。
- 兼容：原 `execute_skill`、`repair_skill`、`refresh_evidence` 等保留。

开放运动只支持局部受限提案（≤0.15 m、≤30°）。父进程核对起始编码器/TCP/附件、证据版本和资源，并检查当前 RGB-D 扫掠样本、实际 IK 关节路径样本、收敛和 F/T。未知扫掠空间拒绝。已记录为闭爪前受控停止且新鲜夹爪开度≥75 mm 时，可以提出空爪脱离运动；这不是许可，仍需通过完整路径与力保护，成功后才清除停止阶段。该实现不是经完整认证的全局规划器：采样视觉网格不能证明连续碰撞安全，缺少可靠可见性时可能拒绝所有横移。无裸 `open_gripper`；释放仍由原 Skill 验证器授权。

## 记忆与审计

`BeliefState` 索引现有公开估计，独立于条件 TRUE/FALSE/UNKNOWN。记录对象、区域部分几何、抓持注册、模型形状、机器人/力觉、质量、版本与来源。旧证据转为历史，不自动成为当前许可；历史原始点云通过原观测归档引用。未知误差维持未知。

容器补证：先使用语义几何；缺测时用历史已拟合壁面关联当前深度，同相机变化检测，再按原阈值重拟合当前法向合格点。它是历史引导的**当前测量**，不是将旧边界当作真值。深度缺失、壁面矛盾、标定改变仍拒绝。

Harness 记录接口返回、效果观测与效果验证；夹爪返回不等于抓持成立。进展含条件、几何覆盖/残差、目标误差与有限有证据搜索，微小 TCP 移动和新 ID 不算进展。完整历史按需查询，恢复上下文只提供当前决策信息。MuJoCo 在 LLM 请求期间不持续步进；不据此声称真机长时间持物安全。

## 独立审计包

```bash
.venv/bin/python -B scripts/local/export_audit_bundle.py \
  --records outputs/model_comparison/本轮目录 \
  --output outputs/audit/capx-v34-$(date +%Y%m%d_%H%M%S).tar.gz
```

打包当前工作树源码（包含未提交/未跟踪源码）、选定实验与 manifest。排除密钥文件、模型、虚拟环境及无关旧数据；文本鉴权字段脱敏。修改过的导出日志保留原内容哈希及导出内容哈希，不伪造原事件链。环境依赖和模型不包含在源码包中。

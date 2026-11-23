# V26 真实模型 + ContractGraph 开发实验

使用正式项目 `.venv` 和已有配置。该入口实际调用 DeepSeek，不会回退到参考程序。

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
RUN="$PWD/outputs/model_comparison/v26_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python -B -m capx.experiments.contract_models \
  --config env_configs/graph_feedback/development-v26.json \
  --seed 0 --output "$RUN/normal"
```

每个新目录生成一次初始 Python，G0/G1 共用该程序；初始完整物理 Snapshot 必须完全相同。每组最多 4 个程序块，恢复由真实 DeepSeek 生成。请求、费用、墙钟时间、物理调用和无进展恢复均有限额。不要求全部 Vision Skill 成功才开始实验。

证据过期压力测试及消融（独立于正常分布结果）：

```bash
.venv/bin/python -B -m capx.experiments.contract_models \
  --config env_configs/graph_feedback/development-v26.json \
  --seed 0 --study evidence_expiry \
  --ablations no_selective no_future no_memory \
  --output "$RUN/evidence-expiry"
```

压力测试仅在一次子目标完成后使其证据过期，不改变物理状态。故障设置保存在评估侧，在线执行器只看到有效性变化。它验证补证/续接，不代表机械抓持失败恢复。

父进程提供 `execute_skill`、`execution_state`、`refresh_evidence`、`repair_skill`。模型不能提供真值、释放许可或任意技能参数。低层运动/释放安全门、Sandbox 和原生物理评分保留。

G0 保守复核未执行后缀与已完成目标；G1 根据选择的契约缩小补证范围，并提供未来意图提案。两组拥有相同 Skill、实际感知、重试能力及限制。`no_selective` 关闭选择性补证；`no_future` 关闭下一工作图提案；`no_memory` 强制重新验证原本会保留的目标证据。这是受约束能力消融，不能解释为删除所有低层几何记忆。

输出包含源代码归档、完整请求/响应/回执、每个 actor 的代码和反馈、物理 Trace、公开 RGB-D、视频、图事件与独立评分。`request-ledger.json` 是去重费用依据；每组报告引用共享初始请求，跨组不能直接相加。价格使用项目现有保守估算口径，非服务商账单。

现有 B0 与 cube G0/G1 入口仍为 `python -m capx.experiments.graph_v2`；新长程入口不把另一套 Skill baseline 冒称原始 B0。详细实测范围和限制见 V26 报告。

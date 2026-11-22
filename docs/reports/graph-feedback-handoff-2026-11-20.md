# 图反馈工作续接断点

正式目录 `/home/hanyu/dqb/cap-x-20261106`，分支 `feature/graph-complete-20261120`，HEAD `823b878e2dc010e3ec0f3683c9f846ffb97fa17c`。保留大量既有未提交文件；不要清理、reset或迁回旧工作区。用户本轮明确真实DeepSeek额度充足，但最后报告 **Codex聊天额度不足**，因此停止执行并保存断点。

## 当前状态

完整开发和正式评测**尚未完成**。最终20新场景/任务×2生成×3组开发(360episodes)和100独立新场景/任务×2×3正式(1800episodes)均未启动。不能用Mock/Oracle/少数历史场景声称图改善成功率。

已保留原评价和Sandbox；共同底座包括有来源ID的目标/TCP换算、抓持偏移、跨actor阶段/预算、释放前检查、精确控制、公开RGB-D桌面/自体分割、图失效/诊断/恢复。G0保守、G1选择性；B0合法原Vision接口。Graph campaign支持冻结/断点/完整性、场景聚类统计及full-G1对单机制消融直接比较。

最近修复：G0普通读不误废弃历史事实衍生的新证据；桌面region意图与边界许可分离；当前帧在旧证据失效后登记；渲染真实MSAA深度采样校准、GL裁剪约定与sample-mask恢复；首次模型请求前公开校准预检。具体见 `graph-development-closure-2026-11-20.md` 与 `outputs/reproduction/graph-complete-20261120/depth-runtime-fix-v10/README.md`。

## 测试

- `graph-unified-v10.log/xml`：990 passed、6 skipped。6个真实渲染opt-in用例分别在Stack/Lift/Restack运行，各6 passed（见depth-runtime-fix-v10/configured-*.log/xml）。
- `p0-online-regression-v9.log/xml`：68 passed、0 skipped，在线视觉Seed重放/Oracle计分/视频保存。
- 原 physical_evaluation.py、audited_stack.py、multitask/runtime.py 与本轮baseline.json保护哈希一致。
- 服务仍复用SAM3 18114、GraspNet 8215、CPU PyRoKi 41337，以development.json为准。NVML/驱动不匹配，已有服务可用；不重启其他服务，不改依赖。执行必须先configure_services加载配置的Mesa EGL/JAX CPU环境，common.sh单独不足。

## 真实开发结果与最后中断

产物根 `outputs/reproduction/graph-complete-20261120`。
- Lift0 v4：B0/G0/G1物理成功，无重规划；属于旧版本开发证据。
- Restack21 v7：B0成功，G0/G1失败；v8 G0/G1均8块耗尽。保留全部失败。
- Restack21 v9：B0成功；G0/G1因DepthCalibrationUnavailable执行前失败，已定位并修复GL运行时探针。0动作的错误不能归因于模型。
- **Restack21 v10最后中断**：仅B0开始；3次模型请求全received/HTTP200/stop，无未知started请求；41物理API，2次重规划，最后完成评价reward0，最终interrupted_by_operator/policy_successfalse。源码369文件终态未变，退出码130。G0/G1未开始；没有完整pair/summary。`episodes/B0/result.json`、视频、Trace、request-ledger.json保留。不要自动resume该单pair或冒充完整配对；先读中断block与Trace。

## 下次最小行动

1. 读本文件、开发闭环报告、`docs/feedback/graph-v2.md`；确认Git状态、无残留本轮实验进程、服务状态。不要先扩大评测。
2. 所有源码目前停止编辑。先用新目录重跑Restack21真实配对，仍用v7同一Snapshot：

```bash
cd /home/hanyu/dqb/cap-x-20261106
bash scripts/local/graph_experiment.sh pair \
  --config env_configs/graph_feedback/development.json \
  --task restack --seed 21 --groups B0 G0 G1 \
  --snapshot outputs/reproduction/graph-complete-20261120/deepseek-restack21-v7/scene/snapshots/seed-21.json \
  --output "outputs/reproduction/graph-complete-20261120/restack21_resume_$(date +%Y%m%d_%H%M%S)"
```

这条命令会产生DeepSeek费用，不会自动续接中断的物理世界。开发重跑与旧结果分版本保留。

3. 核对绿块桌面放置/释放/新撤退→红块抓取/堆叠/释放/撤退；任何共用底座问题先零LLM复现修复。原物理判据不变。线上只公开感知/机器人状态，真值只离线。
4. 再跑Stack/Lift正常与适用故障（support_refresh/target_shift仅有放置的任务，missing_baseline可用于Lift）。实际物理恢复与hold probe等证据采集分开，需真实DeepSeek成功证据。
5. 三个未见开发Snapshot已准备、未进行模型执行：`closure-scenes-v9/{stack,lift,restack}/snapshots/seed-{202611201,202611202,202611203}.json`（按任务对应）。
6. 共同底座和预算稳定后才启动最终20新场景/任务开发；完整操作草案已复制在本报告旁。正式测试前冻结源码/提示词/服务/评分/预算/分析。G0/G1共享每次初始代码和Snapshot，B0独立合法接口，失败和受控/预算退出全保留。正式集不用来调参。

正式主比较G1−G0；G1−B0区分完整系统与共同底座。每场景2次生成按场景聚类，不当作独立样本。normal、stress、三种图机制消融分开。正式当前仅规划，不宣称已完成。

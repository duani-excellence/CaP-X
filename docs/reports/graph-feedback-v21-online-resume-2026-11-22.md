# Graph V21：DeepSeek 恢复后的在线验收（2026-11-22）

正式目录 `/home/hanyu/dqb/cap-x-20261106`，分支 `feature/graph-complete-20261120`，HEAD `f1ecacb4fb0b1e5b9f56ddc2bbbd796f83e023e4`。原有未提交工作保留。本轮没有提交或推送新改动。

DeepSeek 已恢复，本轮在线试验和本地回归已结束，当前实现为 V21.3。**正式评测仍 NO-GO**：Restack 尚缺本版的任务成功证据，完整开发集与图消融尚未完成。没有启动正式测试集，也未启动新的 360 episode 开发批次。历史 402、旧失败与新结果分别保存。

## 已完成的 V21.1 真实实验

目录：`/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/graph_v21_online_20261122_154720`。所有初始/恢复程序均经现有 DeepSeek provider 获取；G0/G1 每对共享初始程序与 Snapshot，B0 使用其合法原接口。真实 SAM3、RGB-D、GraspNet、PyRoKi、Sandbox 和原评分均启用。模型 `deepseek-v4-pro`，reasoning low、max_tokens 32768，原 16 blocks / 192 API / 2400秒预算不变；干预限于 API 边界。

|场景/条件|B0|G0|G1|G0/G1 LLM重规划|
|---|---|---|---|---|
|Stack 953265258 正常|未运行|成功|成功|0 / 0|
|同场景 support_refresh|不适用|成功|成功|0 / 0|
|同场景 missing_baseline|不适用|成功|成功|0 / 0|
|同场景 target_shift|不适用|成功|成功|0 / 0|
|Lift 1003909883 正常|成功|成功|成功|0 / 0|
|Restack 677061216 正常|失败|受控退出|受控退出|5 / 6|

合计 14 episodes、11成功、3失败，但只有 **3个独立场景**。同一 Stack 的四种条件不能当成四个独立场景。不同条件各自生成初始程序，只有每个条件内部的G0/G1共享；这些结果不适于估计正式总体效果或因果成功率提升。

- Stack 正常程序把一个来源读得过早，旧句柄被阻断，自动生成新计划继续；不是完全没有恢复。Lift 两个监控组没有自动配方/LLM恢复，是正常完成例。
- support_refresh：两组新放置都返回并完成。G1实际复用3个来源、RGB-D采集64→61；物理/工具API均13。没有减少SAM3或GraspNet调用。
- missing_baseline：两组通过 `verify_hold` 共动探测，再放置。没有重复闭爪或反复lift；API均16、采集均72。
- target_shift：旧来源及多段运动的当前支撑证据检查阻断后缀；两次有界配方后成功。API均18，G0/G1采集92/90。G1中未返回计划的来源选择不计作实际成功复用。
- Restack：B0经一次真实恢复后声称FINISH，原reward仍0，保留task_failure。G0/G1均在绿块桌面放置阶段遇到证据缺失，最终 `held_surface_evidence_unrecoverable` 退出，reward与策略成功均0。完整失败和预算消耗保留。

G1−G0所列配对全部同成同败，未显示成功率提升。正常Stack实际复用2个来源、少2次采集；支撑刷新少3次采集，这是可解释的证据获取收益，不能扩大为物理控制或LLM调用收益。Restack相反暴露图上下文成本：G0完成的恢复请求输入约17–22K Token，G1约28–31K；G0/G1端到端339.4/590.8秒（单次，包含各自计入的共享初始延迟）。

独立离线标注覆盖内：154正确放行、22正确阻断、12误阻断、6未知、0观察到错误放行。标签只覆盖支持独立物理标注的检查，桌面/释放标签是几何风险代理，不能称全系统准确率，也不能证明反事实释放安全。真值没有进入在线反馈。

20个唯一新请求均HTTP200，无自动重试；输入276,587、输出89,016（reasoning81,294已包含在输出中）。账本成本上界 **$0.7175982**；沿用项目11月20日价表与缓存Token估算约$0.32385–0.64769，非账单。完整收据与逐项诊断：`v21-1-complete-analysis.json`。GPU NVML仍报告驱动/库版本不匹配，未取得显存数据；沿用Mesa/EGL与现有服务，没有重启其他项目进程。

## V21.2 的共享底座修复

在上述完整配对结束后修改 `capx/execution/grounded_recovery.py`，先保存4项失败测试，再修复：

1. 桌面放置没有support对象，missing证据也为空时，旧代码 `missing == support` 将 `None == None` 误当遮挡，连续两份配方各执行一次80mm抬升。现在必须存在明确support对象且命中其缺失证据才生成该观测动作。
2. 已有桌面区域需复核时，旧配方忽略next_action而重新选区域。现在复用现有 `reobserve_table(region_id, change_view=True)`，历史区域只是意图，返回新验证来源才可规划。可能发生视角运动时，随后重新获取held/robot句柄。

两个修复共同用于G0/G1，未改变视觉阈值、物理评分、Sandbox边界或依赖。新测试 `tests/graph_v2/test_table_recovery_v21.py`；配置 `env_configs/graph_feedback/development-v21-2.json` 保留原预算。

新产物目录：`/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/graph_v21_2_online_20261122_161558`。同初始程序、同Snapshot、真实Vision的零新LLM诊断 `replay-restack/` 中，两组都去除了原先两次实际lift，自动恢复返回API由每份2次降为1次；第二份正确调用区域复核与受限视角动作。但持物可见性仍不足，接着需要LLM；此诊断入口禁止新请求，所以显式budget_exceeded退出，不计真实任务成功，也不计一次新的DeepSeek闭环。它只证明多余动作被去除，未证明解决了Restack。

回归：针对性23通过；全图1098通过/6 opt-in跳过，六项真实渲染/Snapshot补验全部通过，去重后 **1104个图测试通过**；原分层28通过。完整JUnit、日志见 `test-summary.json` 中路径。7个受保护评分/环境/Sandbox/actor文件与V20冻结哈希一致。密钥文件仍0600，已检查已完成新产物，未发现配置中的Key泄漏。

## V21.2 新在线验收

固定开发集的新场景与V21.1分开统计，不能把不同场景结果称为版本修复的配对效果。

|任务/Seed|B0|G0|G1|
|---|---|---|---|
|Stack 1257104511|任务失败（1次LLM恢复）|成功（1次自动配方）|成功（1次自动配方）|
|Lift 789990565|成功|成功（1次自动配方）|成功（1次自动配方）|
|Restack 1482414119|系统错误退出，reward0|未执行程序|未执行程序|

9条记录中7条实际执行程序、5条任务成功；Restack两个未执行记录保留，但不能称为两次物理任务失败。详情 `/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/graph_v21_2_online_20261122_161558/final-online-analysis.json`。

Restack B0进一步暴露两处工程问题：

- 底层Robosuite已到episode horizon，外层计数因reset settling后清零而滞后，Runner继续把 `executing action in terminated episode` 当普通代码错误交给LLM。原始失败、无效后续请求和耗时全部保留。
- 第7次恢复请求在审计回调中因null元数据触发 `.get` 异常，尚未记录HTTP元信息。独立 `requests.jsonl` 证明该请求实际HTTP200、141.211秒，**不是402**；但原始响应内容/usage没有保存，不能补造。具体是body还是usage为null已无法复原。补充证据见 `request-reconciliation.json`，原uncertain账本不改写、不重试。预算器阻止了G0/G1的新请求，因此没有完成这两条机器人执行。

该版本13个唯一新请求，12个有已知usage，另1个保守预留。已知输入20,586/输出61,098；账本上界$0.4102098，其中未知请求预留$0.1410882，均不是实际账单。

## V21.3 生命周期与审计修复

完整配对记录结束后才改源码：

- `capx/experiments/graph_v2.py`：检查父进程持有的原生runtime终止状态，覆盖step返回及settling期间抛异常两条路径；停止后续actor/模型恢复，保留已完成的原始reward。该标志来自Robosuite的horizon生命周期，代码中不是任务success条件；不提供给Student，也不根据可伪造的stderr判定。原评分实现未改。
- `capx/experiments/layered_feedback.py`：先记录attempt、HTTP状态、请求ID，再解析body/usage；null、非对象及无效Token计数保留原始响应，账本采用未知usage与保守费用，不能被误计为已知零费用。没有增加重试或替换模型。
- 新测试 `test_simulator_lifecycle.py`（先6失败/1通过）、`test_model_audit_null.py`（先6失败）；针对性85项通过。最后补齐无效usage的归一化断言，先3失败/3通过，再6通过。

产物：`/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/graph_v21_3_validation_20261122_164101`。真实仿真生命周期集成测试将临时环境原生horizon设为当前步数+2（这是测试资源上限，不是算法几何补偿）：原生14/14、外层2/1500时立即 `truncated / simulator_episode_terminated`，没有追加重规划，reward0。**模型为Mock、付费请求0，单独标识，不作为DeepSeek或机器人任务成功。** 见 `native-lifecycle-result.json`。

随后新的真实DeepSeek+Vision Stack953265258配对 **G0/G1均成功**，共1个新请求，HTTP200、输入2,251/输出3,604，账本上界$0.01724316。每组源码后检查370文件未变，见 `online-analysis.json`。最后仅进一步规范异常Token计数的账本usage字段，6项针对性补验通过；没有再发请求，该防御分支不改变正常响应、控制或恢复行为。最终字节清单见 `execution-version-final.json`，不是正式评测冻结声明。

全量图测试1111通过/6 opt-in跳过；复用V21.2已通过且源码未变的6项真实渲染测试，并替换最后6项usage补验，**1117个不同图测试通过**；原分层28项通过。原评分/环境/Sandbox/actor7个受保护文件哈希未变，最终Key扫描无匹配，配置权限0600。

本次恢复额度后共 **34个真实新请求**，33个有已知usage，1个响应内容/usage未知但有独立HTTP200证据。已知输入299,424、输出153,718（reasoning138,494包含在输出内），合计保守账本上界 **$1.14505116**。三版本合计25条试验记录、其中23条执行了程序；涉及6个不同场景，**不合并为一个成功率/置信区间**。历史初始程序重放与Mock生命周期测试均不计入这些真实新请求。


## 复跑与正式评测条件

单对真实付费验收（新的输出目录）：

```bash
cd /home/hanyu/dqb/cap-x-20261106
bash scripts/local/graph_experiment.sh pair \
  --config env_configs/graph_feedback/development-v21-3.json \
  --task stack --seed 1257104511 --groups B0 G0 G1 \
  --snapshot outputs/model_comparison/graph_final_v20_20261122/development-snapshots/scenes/stack-1257104511/prepared-snapshot.json \
  --output "outputs/model_comparison/graph_v21_2_pair_$(date +%Y%m%d_%H%M%S)"
```

当前正式集未动。需要补齐Restack及尚未在线验收的抓持丢失重抓、已释放后撤退失效案例，确认公开感知下可恢复范围；然后冻结通过验收版本、完成原固定60开发场景及消融，按场景聚类评估收益和成本。当前个例只证明动作准入、局部恢复和部分证据复用；图未显示成功率优势，Restack还增加上下文开销。不得据此扩大为正式效果结论。

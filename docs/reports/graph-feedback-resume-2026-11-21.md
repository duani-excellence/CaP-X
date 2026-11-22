# 图反馈续跑与恢复缺口修复

正式目录 `/home/hanyu/dqb/cap-x-20261106`，分支 `feature/graph-complete-20261120`，HEAD `823b878e2dc010e3ec0f3683c9f846ffb97fa17c`。工作区既有修改保留；运行版本由每次产物的完整源码归档标识。尚处开发阶段；V20固定开发360 episodes已启动，正式1800 episodes未启动。

<!-- current-status-start -->
## 当前状态（2026-11-22，V20压力验收完成）

V20新增动作前空夹爪拦截，G0/G1共用。1081个不同图相关测试（含6个真实渲染）及28个原分层反馈测试通过。两项真实DeepSeek＋Vision压力配对均完成，4个episode物理与策略成功，11个去重请求，费用上界估计$0.40590660。定位更新两组各2次重规划/14实际API；目标偏移G0为2次、G1为3次重规划，均17实际API。机制有效，但这组没有证明图降低成本或提高成功率。

V19两个Restack场景结果仍保留：G0 2/2、G1 1/2，不能跨版本合并成冻结效果估计。`outputs/model_comparison/graph_final_v20_20261122/`的60个新开发Snapshot（20/task）已全部重放通过；430项来源及配置已冻结，360-episode开发批次已在受管会话2536实际运行。正式1800 episodes未开始，正式评测仍NO-GO，完整开发数据不足。

首个运行检查点为2/120 units、6/360 episodes完成，单个新Stack场景两次初始生成下三组均成功；仅作运行检查，不是完整效果估计。完整本轮说明见`docs/reports/graph-feedback-v20-development-2026-11-22.md`。原物理评分、Sandbox、依赖和服务未改，未重启其他进程。NVML显存读取问题仍在。
<!-- current-status-end -->

## 续跑状态

复用SAM3/GraspNet/CPU PyRoKi：Schema均200，GraspNet支持Seed。未重启任何服务。NVML/驱动问题仍使GPU显存无法读取；不宣称驱动已恢复。上次v10是人工中断的非完整配对，未直接重放其在途物理世界；v11恢复同一v7 Snapshot，在新目录执行。

## 修复前真实配对 v11

`outputs/reproduction/graph-complete-20261120/deepseek-restack21-v11/readout-v1/`保存完整离线汇总。

|组|原物理reward|策略成功|重规划|实际API（含感知）|实际秒|含共享生成的配对秒|
|---|---:|---|---:|---:|---:|---:|
|B0|0|否|3|44|359.41|359.41|
|G0|0|否|7|16|485.14|485.14|
|G1|0|否|7|20|419.33|515.81|

B0程序正常返回后两块都在桌面上，第一次结束水平中心相差约4.7cm；第一次恢复只撤退，第二次又搬动绿块并沿用旧抓取位置，最终FINISH不计成功。G0/G1共享初始代码遗漏显式open_gripper，正确触发retreat_requires_release，持物阶段没有被新actor重置。后续均进入桌面证据不足循环：G0代码块耗尽；G1持物表面证据不可恢复退出。所有退出均不计成功。

共19次实际请求，输入350,531、输出82,947 token，reasoning包含于输出。现有价目费用上界估计$0.79117104，非账单。不能从这一个历史场景宣称图有成功率收益。

## 最小共同底座修复 v12

仅修改 `capx/execution/grounded_feedback.py`，新增 `tests/graph_v2/test_nonheld_recovery.py`：

1. 持物期间允许从当前公开观测补建**非持物**几何；仍禁止用遮挡中的持物重新标定其尺寸。
2. 桌面遮挡且缺少非持物几何来源时，明确返回缺失对象与“补观测后无运动重验证同一位置”的恢复指引；不将真实障碍改判为缺失观测，不保证补观测一定可恢复。
3. 仅将白名单中的恢复前置条件传入两组模型上下文；图仍分别决定诊断、失效及后缀范围。
4. 明确place/table_place只执行运动，必须显式open_gripper，再创建新撤退计划。没有改成自动释放。

测试先失败后修复：4 failed/3 passed → 新增9项全通过；相关38项通过；图回归997 passed/6 skipped；原分层反馈28 passed。6项为opt-in实际渲染测试，本次未运行，历史三任务单独验收见前续接报告。一次合并测试因同名test_analysis.py收集冲突失败，日志保留，分目录重跑，没有改断言。

保护哈希证明原物理评分三文件与本轮baseline一致。未改Sandbox边界、依赖、服务配置、模型预算；无训练、模型下载或真实机器人操作。修复前源码、diff、JUnit及日志保存于 `outputs/reproduction/graph-complete-20261120/nonheld-recovery-fix-v12/`。

## 冻结公开帧诊断的限制

同一v11 RGB-D、既有SAM3/机器人掩码，0新模型/视觉请求、0仿真步。补齐红块公开几何后，桌面仍有4个未知网格，验证保持UNCERTAIN；已验证机器人投影和持物投影高度截断均未解释剩余缺口，没有据此放宽在线判据。

最初robot-shadow诊断的离线plane取样范围过宽，单独文件保留；后续cutoff诊断修正为生产一致的区域内取样，结果4个未知格，不用早期2格读数。两者都没有改变运行许可。

## 修复后真实验收

v12在同一Snapshot以新DeepSeek初始程序跑G0/G1共享配对，产物 `outputs/reproduction/graph-complete-20261120/deepseek-restack21-v12/`。本轮已完整收尾：

|组|原物理reward/策略成功|重规划|实际API（含感知）|实际秒/配对秒|退出|
|---|---|---:|---:|---|---|
|G0|0/否|3|16|290.65/290.65|unknown_physical_effect_terminal_stop|
|G1|0/否|5|13|321.52/425.52|held_surface_evidence_unrecoverable|

共9次实际请求，输入144,371、输出47,820 token，费用上界估计$0.37993692，非账单。完整读数保存在 `deepseek-restack21-v12/readout-20261121T143708Z/`；历史 v12 不改写成后续修复结果。

G0确实补建了持物期间的非持物红块几何，但桌面仍有未解释的单像素高点，保持UNCERTAIN。后续 IK 在首段运动前拒绝目标，被旧 audited API 包装成ConnectionError，监控未识别其类型化 cause，误判未知物理影响而终止。G1共享初始程序使用Sandbox禁止的_id名字；恢复第一轮再次使用同名，额外消耗请求，最终因持物表面证据不可恢复退出。两项属于待修复共同底座问题，不能作为图效果结论。


## V13 边界异常修复与完整结果

新增 `tests/graph_v2/test_broker_error_feedback.py`：修改前8 failed/10 passed。修改后与原前运动拒绝、失败边界测试合计58 passed。

- 只解包ConnectionError的可信IKConvergenceError类型化cause，复用原单段/成功执行计数与当前段未执行证据。字符串伪装、无证据和第二段失败仍保留未知影响终止；不修改原audited评分/Trace包装器。
- 对policy_violation/invalid_program使用同一Sandbox静态验证器重新检查模型自有代码，给恢复模型提供有界错误位置/原因；不转发stderr、鉴权信息或服务异常正文。执行边界没有放宽。
- G0/G1共享这两项基础修复及API说明。v13复用历史Snapshot、新DeepSeek初始程序，真实在线配对已完整收尾。三服务Schema均200、Seed存在，GPU NVML问题仍未解决；没有重启其他项目服务。

源码修改前副本、diff、测试红/绿日志、三份评分文件不变哈希保存在 `boundary-error-fix-v13/`。正式开发/正式评测仍未启动。自动审批曾拒绝固定readout-v1输出名的写入；随后核查无旧readout，使用带时间戳的新目录及独占创建保存分析，没有覆盖旧产物。


V13图回归：1,017 passed / 6 skipped；原分层反馈28 passed。真实配对G0/G1均reward=0、policy_success=false，退出均为held_surface_evidence_unrecoverable；重规划2/3，实际API13/13，实际耗时251.14/217.80秒。6次独立DeepSeek请求，输入116,318、输出30,646 token，费用上界估计$0.27489792。完整读数位于 `deepseek-restack21-v13/readout-20261121-v14analysis/`。

## V14：有界语义重感知

V13保存的公开帧中，两个未知点离已标定机器人可见网格约7.5mm，不应按机器人删除；它们与当帧两个可见面拟合的绿块首个表面相差约0.03mm。原SAM3掩码距这些点2–3像素，超过既有一像素边界完成范围。没有扩大该范围或2mm几何阈值。

使用原SAM3客户端对同一帧做一次由语义掩码范围决定的上下文裁剪请求，真实HTTP一次、无LLM/仿真动作；映射回原图后，通过未修改的几何/桌面判据，冻结帧从UNCERTAIN变PASS。记录位于 `boundary-error-fix-v13/crop-perception-probe/`。这是在线服务处理历史帧的离线诊断，不是当前机器人恢复完成。

随后在G0/G1共同底座接入：仅当前可见双面几何成立、原检查存在少量未解释高点时，最多追加一次裁剪SAM3估计；由父进程预留共享局部恢复预算，跨actor保留。裁剪估计仍须经过原身份、尺寸、首个表面和桌面检查；缺少预算/来源、真实外物或非持物颜色不会放行。原模型B0接口和物理成功判据不变。

新增机制测试先7 failed后通过；相关54 passed、跨actor/重验证等36 passed；完整图回归1,025 passed / 6 skipped。新L0事件按“证据采集”记录，physical_risk_cleared保持unknown，不算物理修复成功。所有原评分保护哈希不变。

V14真实配对已完整收尾。G0/G1均reward=0、held_surface_evidence_unrecoverable；重规划6/3、实际API18/13、实际秒471.91/156.55（G1配对计时213.77秒）。10次去重请求，输入197,724、输出43,853 token，费用上界估计$0.43465356。真实轨迹首次确认IK前运动拒绝正确返回可恢复状态，而非未知物理影响终止；但后续选点器仍反复产生同一未解出的目标。原始数据与只读汇总位于deepseek-restack21-v14/readout-after-pair-v1。仍未达到正式评测GO。


## V15：替代目标历史与三任务新场景

只将有类型化证据的“首段执行前拒绝”记录为当前抓持周期内的失败目标历史，失效其旧region/plan。新observe_table从当前公开点云选择未尝试的位置，保留对真实障碍和覆盖度的验证。未知物理影响仍终止，不把一次IK失败宣称为所有姿态不可达。历史跨actor保留，通过图边绑定新目标。4项机制测试先失败后通过；图回归1,029 passed/6 skipped。

同一版本分别执行此前只准备Snapshot的新开发场景；G0/G1各对共享初始程序。9 episodes全部收尾，源字节检查通过。readout：`new-scenes-v15/readout-complete-20261121/`。

|任务/Seed|B0|G0|G1|说明|
|---|---|---|---|---|
|Stack/202611201|物理与策略成功|物理与策略成功|物理与策略成功|G0/G1均有局部精度修正，无LLM重规划|
|Lift/202611202|物理与策略成功|物理与策略成功|物理与策略成功|三组均无局部恢复/LLM重规划，正常完成|
|Restack/202611203|物理与策略成功|reward0/代码块预算耗尽|reward0/代码块预算耗尽|G0/G1各7次重规划，原失败全部保留|

Restack G0/G1实际API22/12，配对总耗时543.61/410.27秒。9 episodes共20次去重DeepSeek请求，输入343,696、输出76,055 token，费用上界$0.75485652（非账单）。不能把图组实际运行省下的共享生成时间算成算法加速，也不能从此小样本断言图收益。

Stack真实物理精度修正案例：G1第一次修正对应的独立离线TCP误差从6.3095mm降至0.2802mm，公开机器人状态驱动一次真实goto_pose，最终任务成功。该案例证明共同底座可执行物理修正；G0也成功，不证明图独立收益。Lift提供不依赖恢复的正常完成证据。

## V16：当前完整持物侧面定位（开发验收中）

Restack失败帧的旧持物检查只凭完整顶面放行，却未提供当前朝向几何；桌面检查将26个语义边界点视为未知高点。当前公开RGB-D中能观察到一个完整侧面、底边及顶面。复用已有完整侧面拟合器和已公开测量的边长，结合公开相机外侧方向建立当前定位；保留完整边覆盖、平面残差、点云包含和原15mm/10mm一致性阈值。缺相机、侧面不完整、位置矛盾不增加授权。

底边来自当帧顶面减历史公开边长并要求当帧底边像素，明确不是桌面接触证据；同一相机推断也不是独立抓持/滑移验证。已有一像素语义边界及2mm首表面阈值保持不变。

6项测试先失败后通过，相关38项通过。冻结公开帧及旧SAM3结果0新请求/0仿真步验证：原FAIL、26未知点→当前完整面+原语义掩码UNCERTAIN、1未知点→当前完整面+已保存裁剪语义掩码PASS、0未知点。产物`complete-held-face-v16/frozen-public-frame/`。这是离线证据验证，不冒充在线恢复。

新的development-v16.json将G0/G1/B0统一代码块上限设为16；模型32768输出token、192物理API、24局部恢复、8视角、2400秒episode上限不变。120开发units的最大请求5640，设置有限总请求6000；旧配置和失败不覆盖。后续实际结果单独记录，当前仍非正式冻结版本。

V16完整图回归：1,035 passed / 6 skipped / 0 failed（181.39秒），日志/JUnit位于complete-held-face-v16。6项实际渲染opt-in测试未在该pytest命令启用，单独真实在线实验另记。新版真实Restack三组已启动，结果不提前宣称成功。


## V16在线开发中断与V17恢复修复

V16三组开发未完成：B0 reward0，一次恢复后FINISH未改变物理结果；G0在空夹爪后被错误反馈为仍持有绿块，模型随后反复生成放置旧持物的代码。为了修复已确认工程错误，使用现有SIGINT收尾保存结果、图和视频。G0标记interrupted_by_operator，G1未运行；不生成完整summary、不作配对效果分析、不自动续播物理后缀。两组已运行来源检查仍通过。具体请求、费用和未知在途请求见deepseek-restack-new-v16/interruption-audit.json；未知用量不能按0处理。

V16新的冻结帧还有12个未知点：包含机器人边缘和RGB红色/深度前景的边缘歧义。一次真实SAM3裁剪无法消除这些点，没有扩大掩码或放宽身份/深度条件。原来准备的“所有FAIL都裁剪”的修改未应用，诊断及未应用测试保留。

V17改动G0/G1共同底座：
- 当前桌面检查因较多未解释高点FAIL时，把该位置记录为本抓持周期的搜索排除历史；不声称那里有真实障碍或全局不可达。旧区域、放置/释放许可失效，新actor可选择不同的当前可见空区。所有后续运动/释放仍按原验证器执行。
- 原有空夹爪检测成立时，明确清除当前持物、刚体附着、旧放置和基线；历史动作、对象身份和累计预算保留。反馈新增grasp_active/lift_verified/last_grasp_loss，释放后的对象也不再报告为当前持物。有效抓持跨actor保持不变。

机制测试先红：替代区域4 failed/3 passed，抓持生命周期6 failed。首次组合针对性测试31 passed/2 failed，原因为新增测试将初始化get_object_pose也计入总API却只预期一次goto；修正为显式两次调用顺序，未放宽行为断言。各版日志独立保存于table-obstacle-alternatives-v17。

V17针对性复跑33 passed；完整图回归1,048 passed / 6 skipped / 0 failed（182.42秒）。另用实际Restack Snapshot单独补验上述6项渲染/监控测试：6 passed，0模型/感知HTTP、0机器人API。合计覆盖1,054个不同测试。新版DeepSeek G0/G1配对位于deepseek-restack-new-v17，尚未完成时不报告成功。三个原评分文件哈希与v15归档一致。


## 2026-11-22：V17完整配对结果与V18

V17 G0/G1共享初始程序及Snapshot，来源检查通过，均reward0/policy_success=false。G0重规划8次、实际API21次、配对耗时547.94秒，hold_evidence_unrecoverable退出；G1重规划15次、API24次、配对耗时965.13秒，16代码块耗尽。24次去重模型请求，输入462,913、输出96,613（含reasoning90,152）token；费用上界估计$0.99363264，无未知用量。完整只读汇总位于deepseek-restack-new-v17/readout-complete-20261122。

两组真实执行中均记录了旧区域/计划失效、新区域选择，以及滑落后撤销当前抓持许可。G0新actor已根据无持物状态改为重新观察、抓取红块；G1也正确进入重新抓取。该证据证明机制执行，并非恢复成功：最终仍失败。图节点/边与示意图见deepseek-restack-new-v17/mechanism-evidence，不能从单场景宣称图收益。

G0离线评分状态表明红块被夹住且抬起，但在线视野中不可见；旧verify_hold未换视角便耗尽次数。G1在缺失支撑块观测时，恢复代码反复调用会中断actor的接口，误以为能返回None供后续分支处理。

V18采用最小共同修复：在已有公开刚体附着、抬升高度、邻物间隙及剩余预算满足时，verify_hold可先尝试一次已有的原位yaw视角调整；新视图本身不能提升抓持阶段，仍要通过原视觉随动检查。没有新增抬升动作、传感器或物理成功判据。另把缺失物体名与“失败会中断actor而非返回None”加入结构化反馈，提示先规划观察位置再重试支撑块观测；不自动执行额外运动。新增测试先6 failed/5 passed，修复后相关51 passed；完整图回归1,059 passed / 6 skipped / 0 failed（181.72秒），6项实际渲染补验待本轮在线试验结束后执行。


### V18已完成Stack/Lift开发回归（2026-11-22）

同一V18代码下，Stack seed202611201与Lift seed202611202的B0/G0/G1均原reward1且policy_success=true。Stack三组配对端到端耗时42.48/113.95/125.92秒，重规划0/1/1次，实际API11/15/15次；G0/G1各有3次物理TCP修正、2次视角采集。第一次精修并未改善误差，随后才收敛，不能将全部修正计为成功恢复。Lift三组0次重规划，G0/G1各有1次证据采集、0物理修正，因此是无需重规划完成，不能称完全无干预。

6个已完成episode的全局去重请求6次，输入41,295、输出16,092 token，费用上界$0.11823372。完整明细见new-scenes-v18/readout-stack-lift-20261122。G0/G1的同一初始请求只计一次实际费用，配对端到端耗时仍各计入共享初始生成时间。

Stack真实图事件中，G0保守废弃未执行后缀、恢复上下文选完整代码块；G1选择依赖切片并保留有效物体/表面结果。两组各15个returned_call均在最终图保持done，未被重置为待执行。证据见new-scenes-v18/stack/mechanism-evidence/public-graph-decisions.json。这是机制执行证据；该场景双方重规划/API次数和最终成功相同，不支持图效果提升结论。Restack仍在运行，未完成前不作三任务整体成功率或正式GO判断。


### V18 Restack中定位的桌面搜索数据源问题（待配对结束后修复）

冻结G0 block5公开RGB-D重算：原始绿块颜色范围x=[0.10086,0.33717]米，被误分割点拉宽到23.63cm；搜索函数加边距后触发既有最大半宽限制，返回None。相同帧使用既有attachment/可见面验证器得到PASS，保持原table候选/覆盖/障碍阈值不变即可找到(0.45302,0.03928)米的候选、2217个可见平面点、覆盖率1.0。该结果仅是冻结观测反事实，0模型/视觉请求、0仿真步，不代表线上物理恢复。证据见unverified-hold-view-v18/frozen-table-search-block5/result.json。

拟将observe_table的持物候选几何绑定到既有当前验证结果，并保留历史候选范围与当前运动/释放许可的区分。新增6测试先5 failed/1 passed，缺失当前形状时不得回退旧值，后续table/release检查仍必须执行。草案与RED产物保存在unverified-hold-view-v18/pending-table-search-v19；当前V18运行未结束，尚未应用业务源码修复。另一次阻断的单个夹爪边界像素仍保持UNCERTAIN，未扩大掩码、几何容差或放宽评分。


### V18完成与V19共同修复

V18完整9 episode已经结束，所有组source_postcheck均unchanged=true（每组369文件）。Stack/Lift各三组成功；Restack B0原reward0，3次重规划、49 API、435.40秒，FINISH不改变失败；G0原reward0，15次重规划、20 API、914.87秒，耗尽16代码块；G1原reward0，9次重规划、28 API、460.49配对秒，hold_evidence_unrecoverable受控退出。G1实际清除了滑落绿块的当前抓持状态并重新抓取红块，但红块随动证据仍无法确认，因此不能称恢复成功。

全9例去重模型请求35，输入580,209、输出154,820 token，费用上界$1.37896308。结果位于new-scenes-v18/readout-complete-20261122。35个请求均有receipt，其他历史版本的未知请求不包含在这一费用内。V18补验真实渲染6 passed（5.72秒），与图回归1,059项合计覆盖1,065个不同测试；不是将skipped冒充passed。

V18停止且来源核验完成后，已应用V19桌面搜索几何绑定。只修改grounded_feedback.py中该候选搜索路径及其API文档，新增test_table_search_shape.py与development-v19.json；候选生成仍调用原choose_table_region，运动/释放仍调用原_current_table_check，三个独立评分文件哈希未变。先失败5/6，新实现相关118 passed，原L0–L3反馈28 passed。完整图回归1,065 passed / 6 skipped / 0 failed（180.96秒）；随后在V19实际Restack渲染环境单独补验6 passed（5.66秒），合计1,071个不同图相关测试。

实际冻结公开观测通过新版G0/G1 observe_table接口，两者均返回(0.45302,0.03928)米候选，fit_check=PASS、shape fact permission=false，并保留图来源。该API重放0模型请求、0感知请求、0broker调用、0仿真步；见table-search-shape-v19/frozen-monitor-api。候选的历史形状依据不等于当前释放许可。夹爪边界单像素离线与公开机器人first-ray几何相符，但语义有冲突，线上仍保持保护和UNCERTAIN，未更改该过滤策略。


V19真实DeepSeek G0/G1回归已启动：先deepseek-restack-new-v19（seed202611203，同历史Snapshot），再deepseek-restack-unseen-v19（seed202611204，新生成可重放Snapshot）。每对共享初始程序、Snapshot、预算与评分，仍属于开发验收，不是正式效果评估。运行期间源码保持不变。

allocation-candidate-v19保存20/task开发、100/task候选正式场景的独立分配及历史检查：1101份当前metadata加既有历史审计，322个已用Seed，候选360个Seed无交集、无JSON解析错误。没有准备或执行这些场景，最终冻结前仍需再次核对新增历史。


### V19首对真实Restack结果（完整）

G0原物理reward1、policy_success=true，10次重规划、35 API、537.46配对秒；G1原reward0、policy_success=false，9次重规划、27 API、457.58配对秒，以hold_evidence_unrecoverable退出。每对初始程序和Snapshot相同。此次G0胜，不能据此主张图成功率提升。20个去重模型请求均有receipt，输入465,202、输出72,947 token，费用上界$0.90293676。

G0轨迹包含：桌面检查阻断→选择替代区域→绿块滑落、清除持物状态→重新抓取红块→放置处持物观测不可确认→DeepSeek生成抬高5cm的观察动作、更新持物/支撑表面及TCP→重新放置、通过原释放前验证、撤退→独立评分成功。最后程序为真实模型恢复代码，非Oracle。该成功说明共同底座的异常阻断/恢复链可工作，不是图增益证据；物理修正4次、证据采集12次的逐次成败均保留。离线检测50个eligible检查点中47可独立标注、3未知，不将未知或仅视觉PASS记作正确。

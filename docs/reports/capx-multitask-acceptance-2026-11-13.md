# CaP-X 多任务与前端同步验收 — 2026-11-13

正式目录 `/home/hanyu/dqb/cap-x-20261106`，分支 `feature/teacher-data-20261112`，HEAD `823b878e2dc010e3ec0f3683c9f846ffb97fa17c`。未提交本轮修改；保留开始时已有的未提交工作。所有实现及实验在正式目录，未在旧压缩包工作区开发。

## 结果

- **已实测可采集（5 个不同具身任务）**：Stack、Lift、Restack、Nut Assembly、Two-arm Handover。新增四项均在 Seed 0、21 实际通过 Oracle；Stack 复用原有 P0，在本轮 Seed 21 Oracle 和 Seed 0 采集链路回归通过。
- **已接入但未验收（2 个）**：Wipe 的 Seed 0 未擦完所有污渍；Two-arm Lift 的 Seed 0 未抓稳双把手，Seed 21 成功。均禁止付费采集，不因一个 Seed 或 Mock 通过就放行。
- **仍有阻塞**：LIBERO 打开微波炉/抓取汤罐已有适配，但导入实际失败：`No module named robosuite.environments.manipulation.single_arm_env`。当前 LIBERO 依赖旧 Robosuite 接口；未改主环境。R1Pro/BEHAVIOR 的 OmniGibson、BDDL 依赖未安装，未另搭重型平台。
- 未运行付费 Teacher 请求、大规模采集、训练、真实机器人，未下载模型或升级依赖。

## 实现范围

新增 `capx/multitask/{catalog,runtime,oracles,worker,cli,stack_oracle}.py`、`scripts/local/multitask_teacher.sh`、`capx/web/multitask.py`、`tests/multitask/`、`docs/multitask/`。修改 Web 路由、前端 `ConfigStartControl.tsx` 并新增 `MultitaskControl.tsx`；README 加入当前入口。

Stack 使用原有 pilot/Teacher/campaign/Sandbox，不替换它的奖励定义。新增任务通过独立进程的适配模块复用现有低层环境、API、模型客户端、代码解析、Sandbox、采集检查点及导出筛选。未修改原有 Teacher/campaign 源文件，因此不主动破坏已有 Stack 检查点签名。

不同任务使用各自 Robosuite `_check_success()`，并加持稳/释放/接触判据；不用 Stack 的评分套全部任务。新增任务完整 reset→settle 历史在新环境重建，核对 model、physics、RNG、ctrl、warmstart 及状态 hash；并非声称可恢复任意中途 Snapshot。每个场景先 Oracle 后 Teacher，二者必须从同一已保存 Snapshot 重建。

隔离沿用 Bubblewrap/SandboxProcess，不放开 scipy、globals、env 或 obs。原 Oracle 的 scipy 旋转转换为 NumPy 实现，修正 Restack 半尺寸计算、Lift/Restack 位姿 WXYZ 与实际物体尺度，以及单臂初始关节缓存。双臂视频补充仿真时间戳。超时/系统错误后环境失效，后续 trial 新进程重建。

Trace 保存每次 API 参数、返回值及前后物理状态（物体位姿/速度、EE 位置、夹爪开度、接触和任务成功状态），同时有逐步物理记录。新增通用 trace 尚未统一补充 EE 朝向和完整相对位姿，列为后续完善项，不声称与所有历史 P0 trace 字段完全相同。

## 本地测试证据

产物根：`outputs/reproduction/multitask-20261113/`，全部使用新目录。

1. `tests-final.log` / `tests-final.xml`：**133 passed / 2 skipped / 0 failed**，覆盖新增任务、Teacher/campaign、pilot 契约/Token/Web 选项及 P0。
2. `web-jobs-tests.log` / XML：额外 **3 passed**，验证目标转发、CLI 入口、续采预算一致性、未验收拒绝、运行冲突及停止信号。两次合计 **136 passed / 2 skipped / 0 failed**，不是同一条 pytest 命令的统计。
3. 两项 skip：`test_seeded_visual_service_and_video`、`test_paired_oracle_physical_accounting`，本轮未设置视觉服务测试开关；未宣称完整 Vision 回归通过。
4. 新增四个 ready 任务 Seed 0/21：独立 Oracle 物理成功，Snapshot 重建相等，pass/空程序/语法错误/隐藏 env/虚假 FINISH/无限循环超时六项保护检查通过。真实结果分别见 `<task>-v2-seed<seed>/acceptance.json`。
5. `collection-mock/`：五任务各 1 个真实仿真 Mock episode 均物理成功，各任务 HTTP 请求 0、SFT 合格 0；合并导出为空，符合 Mock 不可当 Teacher 数据的约束。
6. 前端 `npm run build` 成功（TypeScript + Vite），日志 `web-build-final.log`。旧 dist 保存在 `web-dist-before/`。
7. 浏览器实际选择 Lift 并执行免费 Oracle，作业 `2d561c18779d40ac8609659e1f5dea4f`，exit 0、Oracle success、Snapshot replay true、六项 guards true、模型请求 0。结果位于 `outputs/multitask-web/<job-id>/run/`。

早期实验错误（Oracle 中不允许的导入、调用预算、函数选择、视频时间戳）均保留原日志，不覆盖失败记录。Web 旧测试曾因未等待后台 coroutine 而出现 IndexError，已改用有界 Event 等待捕获回调，未修改业务行为来迎合测试。

## 采集、导出与前端

每任务独立目标、Seed 序列、累计请求/成本预算、场景/时间上限、检查点、有限重试和去重；复用既有 unknown-attempt 拒绝静默重发逻辑。串行运行任务，一个独立任务失败继续其他任务。无训练 batch size 语义。

合并 SFT 采用等任务配额，保存 task/source/mode/snapshot/split 信息；按任务+场景分组划分，拒绝不完整模型响应、物理失败和 Mock。一个选择任务暂无合格样本时合并导出为空，但每任务数据独立保留。

前端与 CLI 共用任务目录：选择任务、分别设置目标、每任务请求/美元上限，免费 Oracle、单次推理、采集、停止和续采。未验收任务置灰并显示原因。只重启了已确认空闲的 CaP-X Web（新 PID 592368），未停止感知服务或其他 GPU 项目。已刷新 `http://127.0.0.1:8200/`。

多任务面板固定 **DeepSeek + Privileged**，顶部原有模型/观测选择仍用于 Stack 交互推理；面板已明确提示。多任务当前只展示进度和产物路径，不提供原交互会话的实时 Viser 画面。服务器重启后的历史任务需核查进程状态，不能把内存状态丢失视为作业必然结束。

## 尚存问题与边界

- Wipe 需进一步修复接触路径覆盖；Two-arm Lift 需提高双把手抓持稳定性。没有放宽成功判据或挑掉失败 Seed 以冒充通过。
- LIBERO 需要隔离兼容环境，并重新核查物理成功及接口，当前不可采集。
- 新增任务只验收 Privileged；原 Stack Vision/蒸馏保留，但未扩展为未经验证的新任务 Vision。Privileged 数据不能直接宣称部署 Vision Student 可用。
- paid Teacher 质量、更多 Seed 泛化和真实供应商结算费用本轮未测试。ready 仅允许下一步受限小采集。
- 多任务代码/配置被固定进检查点；修改后续采会拒绝，需显式审查迁移，不能简单去掉签名检查。
- 本轮不承诺大规模目标一定可达；同模板上限、预算或 Oracle Gate 可能先停止。

实际单任务、多任务推理、采集、续采和导出命令见 `docs/multitask/README.md`。第一批推荐每个已通过任务目标 2 条、每任务请求上限 10、美元上限 2；整体最多 50 请求槽 / 10 美元记账预算，运行前保留成本限制。

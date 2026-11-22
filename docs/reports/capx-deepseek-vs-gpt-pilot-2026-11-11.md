# CaP-X：DeepSeek V4 Pro vs GPT-6 Astra 五场景配对 Pilot

报告生成：2026-11-12T00:10:09+08:00。实验标识按请求保留 2026-11-11；执行跨越当地午夜。

## 1. 结论与范围

**在本轮 Privileged、单轮、4096 completion token、Provider 默认推理配置下，GPT-6 Astra 5/5 成功，DeepSeek V4 Pro 1/5 成功。** 五个场景的配对输入 hash 全部一致。

- Privileged 模型比较：**GO，本轮完成**；建议按同一预注册协议扩展 20 场景，尚未执行。
- Vision 模型比较：**NO-GO，未调用**。Vision Oracle 只有 2/5，未达到 4/5 Gate。
- 当前证据支持：GPT 在这一预算与默认参数组合下，更可靠地产生完整可执行操作程序。**不支持脱离预算条件给出模型总体能力排名**。
- Teacher 数据：已有独立物理成功样本可作为小规模候选（DeepSeek 1、GPT 5），但**大规模 Teacher 采集 NO-GO**；先完成 20 场景、预算敏感性和数据验收。Vision Teacher 暂缓。

正式矩阵实际完成 20 Trial：Oracle 10 + Privileged 模型 10。因 Gate B 失败，跳过 Vision 模型 10 Trial；没有为了凑 30 次绕过 Gate。额外保留两次早期视频基础设施失败尝试，详见第 10 节；不混入正式矩阵。单元/回归测试不计入上述 Trial 数。

## 2. 环境、提交与保护

唯一项目：/home/hanyu/dqb/cap-x-20261106。
独立分支：experiment/paired-pilot-20261111。
正式 Oracle/model Trial 提交：823b878e2dc010e3ec0f3683c9f846ffb97fa17c。
前置实现提交 b575a30：静止 Snapshot v2、manifest CLI、请求/费用/单代码块预算。
修复提交 823b878：消除实验入口重复初始视频帧，保留视频导出失败帧，Gate 排除基础设施失败。

Python 3.12.13；PyTorch 2.8.0；CUDA 12.8；NumPy 1.26.4；MuJoCo 3.5.0；Robosuite 1.5.1；Transformers 4.57.6。RTX 5090 D v2 24GB。
SAM3 8114、P0 GraspNet 8215、PyRoKi 8116。预检 OpenAPI 通过，Vision Oracle 实际调用严格检查 Seed 回显。
没有运行服务启动管理器、没有加载 8115 副本、没有停止现有或其他项目 GPU 进程；没有启动 Web、下载模型或升级依赖。

P0 Reward、physical_evaluation、Sandbox、actor、RPC protocol 与 P0 原提交逐字节一致。已有 369 个产物和两个私有配置哈希未变，配置仍 0600；新增文本未检出配置中的密钥值。没有删除未跟踪文件，tracked 工作区干净。最终状态/版本/配置 SHA-256 见 analysis/protection.json 与 provenance.json。

测试：完整 P0+初版 Pilot **80 passed**（68+12）；视频缺陷单测先失败，最小修复后 Pilot **13/13 passed**。最终覆盖原 P0 68 项与 Pilot 13 项；没有通过删测试或放宽阈值使其通过。

## 3. 新静止场景与 Snapshot

候选 Seed 0、21、1、2、3 全部合格，无替换、无重复凑数。
reset 后保持原关节目标、打开夹爪，让动力学自然 settle；没有执行抓取/移动 API，也没有把物体速度直接清零。每次检查物体速度、非任务成功、物体分离、有限状态及关节限位。最大 settle 5 秒，超时拒绝；连续线速度 <0.02m/s、角速度 <0.2rad/s 至少0.5仿真秒。

五个场景均经过 11 个保持控制步，在连续0.5秒静止后建立 episode 起点。v2 Snapshot 保存物理数组、RNG、控制目标、model/state hash 和 settle_steps。恢复流程为 reset→重放固定保持步数以重建控制器历史→P0 恢复→核对精确 hash；不是仅 reset(seed)。v1 历史 Snapshot 保持兼容，未改历史文件。

| Seed | 初始物理 hash | 最大线速度 m/s | 最大角速度 rad/s | 连续静止验收 |
|---|---|---:|---:|---|
| 0 | a8abf2d359cdc9469fe4e0756b1e4d0cf4e5d8c3f3dbeae9c66190d10e0c5f11 | 1.377e-10 | 1.191e-15 | 0.500 s / 通过 |
| 21 | 5b1242acf526df4a64d4a7f16e19963edc284ac4938c8542dcde4fa7b5213b6d | 1.377e-10 | 1.191e-15 | 0.500 s / 通过 |
| 1 | c24a67aef0716c442b0f8bafaa013a0c8af3c91205580319bd1c0fc38957a225 | 1.377e-10 | 9.232e-16 | 0.500 s / 通过 |
| 2 | 2a696c79c4ae5214374d855bc90cb30c743b7d03480e30b7a13fd95b26722f3a | 1.377e-10 | 9.974e-16 | 0.500 s / 通过 |
| 3 | 49678746e55dcef862b60c8e089613cba9fb21d62395cf19971e8666f11842cf | 1.377e-10 | 1.033e-15 | 0.500 s / 通过 |

完整初始物体位姿、关节、EE/TCP、接触、速度、model hash、配置 hash、API 版本及逐步 settle 记录见 [scenes.json](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/manifest/scenes.json)；Snapshot 为其同目录 snapshots/ 下五个文件。配置/源码完整校验另见 [provenance.json](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/analysis/provenance.json)。新 Snapshot 不覆盖之前不满足静止阈值的场景。

## 4. Oracle 控制组与 Gate

| Seed | Privileged 成功 | Vision 成功 | Vision 最大抬升 m | Vision 释放后水平偏移 mm |
|---|---:|---:|---:|---:|
| 0 | 1 | 1 | 0.204 | 14.56 |
| 21 | 1 | 0 | 0.196 | 48.83 |
| 1 | 1 | 0 | 0.203 | 52.76 |
| 2 | 1 | 1 | 0.207 | 13.55 |
| 3 | 1 | 0 | 0.206 | 48.99 |

Privileged 5/5：Gate A 通过。Vision 2/5：Gate B 不通过。
两模式均 5/5 建立抓取并抬升，无被评分器认定的滑落；Vision 失败集中在 Seed 21、1、3 的放置。
Vision 物体中心估计与真实中心误差约9–18mm；失败释放后的水平偏移约49–53mm，红方块最终回到桌面（两物体中心 z 差约-5mm），没有稳定叠放。**这是关联诊断，不证明全部误差只由定位造成**；TCP/抓取偏移、控制到位误差和放置过程也可能参与。原历史 Seed21失败记录仍保留。

## 5. 严格预算与公平性

入口：scripts/local/run.py --snapshot-manifest <scenes.json> --pilot-phase prepare/oracle/models --output-dir <新目录>。
本轮正式结果位于 run_02；prepare 使用顶层 manifest/scenes.json，后续显式消费该清单。
每 Trial 新建环境并恢复同一个 Snapshot，核验 hash 和静止条件，关闭后才进入下一个 Trial；进程文件锁禁止该运行目录并发。

每 Trial 一个逻辑模型请求、最多两次 HTTP 尝试，仅429/500/502/503/504或网络连接/超时故障允许一次重试；401/403/404立即失败。200秒请求阶段总 deadline；4096 completion token；一个 AST 可解析 Python 程序；90秒 actor deadline；P0 broker 强制64 API调用。没有后续模型恢复请求。
正式模型调用 **10 次 HTTP、全部200、重试0**。没有重复 Smoke。

两 Provider 各5USD累计上限；调用前按最大输入/输出与两次尝试保守预留，usage 返回后按保守上限入账。保守账本 DeepSeek约0.08558USD、GPT约0.15005USD，均远低于上限。真实 usage 的公开价格估算见第8节。

模型顺序：Seed0 DS→GPT；21 GPT→DS；1 DS→GPT；2 GPT→DS；3 DS→GPT。
同一模式/Seed 两模型的 task/system prompt、API docs、公开机器人观测、receipt schema、Sandbox、Reward、Snapshot、动作/代码/输出预算及超时一致。五对 Prompt/API/观测/Snapshot hash 均相等；[hash清单](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/manifest/prompt_hashes.json)。
Privileged 与 Vision 的 API文档允许不同；Vision 模型不运行，不能计算其模型效果。

不向模型发送原始图像或隐藏物理真值。公共初始观测为机器人关节/笛卡尔状态；模型通过相同 API 获取该模式可获得的信息。真实物理状态只用于可信父进程 Trace/评分/离线诊断。

DeepSeek 实际参数：model、messages、temperature=0.2、max_tokens=4096；未显式覆盖 thinking/effort。
GPT 实际参数：model、messages、max_completion_tokens=4096；不发送 temperature 或 reasoning_effort，使用服务端默认推理设置。
两个 Provider 的默认推理策略不等价，**本 Pilot 比较的是这些已记录的部署设置**。没有单独改 Prompt、增加某一模型预算或替换模型。

## 6. 模型汇总与逐场景配对

| 指标 | DeepSeek V4 Pro | GPT-6 Astra |
|---|---:|---:|
| Privileged Success | 1/5 | 5/5 |
| Vision Success | 未运行（Gate B 未通过） | 未运行（Gate B 未通过） |
| Grasp Success | 2/5 | 5/5 |
| Lift Success | 2/5 | 5/5 |
| Placement Success | 1/5 | 5/5 |
| Code Execution Success | 2/5 | 5/5 |
| 平均代码生成时间 / s | 52.372 | 11.762 |
| 平均执行时间 / s | 0.451 | 1.402 |
| 平均机器人 API 调用 | 3.4 | 10.4 |
| 平均输出 Token（含 reasoning） | 3969.4 | 410.2 |
| 平均 API 费用估算 / USD | 0.00806771 | 0.03001000 |
| 总费用估算 / USD | 0.04033854 | 0.15005000 |
| reasoning tokens 合计 | 19283 | 431 |

所有均以5个场景为分母。无代码样本记0次机器人 API、0执行秒；因此 DeepSeek 平均执行时间更短主要来自未执行样本，不能解释为控制更快。Code Execution Success 只表示 sandbox_rc=0，不表示完整程序或物理成功。

| Seed | DeepSeek reward / 结果 | GPT reward / 结果 | 生成时间 DS/GPT 秒 | 输出 Token DS/GPT | 配对 hash |
|---|---|---|---:|---:|---|
| 0 | 1 / 成功 | 1 / 成功 | 42.71 / 11.62 | 3463 / 423 | 四项一致 |
| 21 | 0 / 4096 Token 耗尽，无代码 | 1 / 成功 | 55.37 / 11.55 | 4096 / 378 | 四项一致 |
| 1 | 0 / 4096 Token 耗尽，无代码 | 1 / 成功 | 56.32 / 10.43 | 4096 / 373 | 四项一致 |
| 2 | 0 / 代码截断，仅抓取/抬升 | 1 / 成功 | 54.52 / 10.71 | 4096 / 403 | 四项一致 |
| 3 | 0 / 4096 Token 耗尽，无代码 | 1 / 成功 | 52.94 / 14.49 | 4096 / 474 | 四项一致 |

DeepSeek：Seed0完整程序成功；21、1、3均 finish_reason=length，4096 completion token 全为reasoning，content为空；Seed2也 length，3776 reasoning +320其他输出token，仅返回到抓取/抬升的语法有效程序，未包含后续搬运/放置。没有追加预算或恢复请求。
GPT：五个程序均 finish_reason=stop、执行成功、物理成功。

## 7. 分阶段物理分析与七个问题

- 抓取：DeepSeek两个实际执行程序、GPT五个程序均建立可信双侧接触并成功抬升。空代码不是抓取决策错误，但必须计入端到端失败。候选/评分文件用于 Vision Oracle，Privileged 使用真值抓取 API，不伪造其 GraspNet候选。
- 抬升：DeepSeek Seed0约0.194m、Seed2约0.219m；GPT约0.196–0.248m。物体相对 EE 变换和接触全过程在 Trace 中；本轮实际抓取样本未被评分器判滑落。
- 搬运：所有 goto 请求、实际FK/TCP、接触和目标残差均已离线转换到同一 world frame 记录。TCP残差并非零，模型 Trial 的最大返回位置残差约56mm；底层运动函数没有显式返回收敛标志，不能用“调用返回”宣称精确到位。不同阶段/目标的残差不可简单当成放置误差。
- 放置：DeepSeek成功样本释放后水平偏移约0.86mm；GPT成功样本约0.89–26.36mm，最终均满足原P0稳定叠放判据。DeepSeek Seed2没有执行释放/放置，不能把开场 open_gripper 当最终释放。Vision三个失败样本的真实释放偏移显著更大。
- 碰撞：原始接触对全部保留。桌面支撑和夹爪接触属于预期接触；本轮未将所有 contact 计为异常碰撞，也没有另造碰撞奖励。

问题回答：
1. 哪个更容易生成可执行程序？本预算下 GPT为5/5，DeepSeek为2/5；其中DeepSeek一份被截断且任务不完整。
2. 哪个抓取决策更准确？实际执行样本都成功抓取，样本太少且DeepSeek有生成失败，不能证明其抓取几何决策更差。
3. 哪个放置更稳定？端到端GPT为5/5，DeepSeek为1/5；但DeepSeek主要没有生成完整放置阶段，不能单纯归因于放置控制质量。
4. 失败模式不同吗？DeepSeek四个失败集中在completion预算耗尽；GPT本轮无失败。视觉Oracle失败属于放置链路，未混作模型失败。
5. GPT额外推理是否带来物理收益？本轮反而DeepSeek记录更多reasoning tokens（19283 vs431）；不能支持“GPT多花推理所以更成功”的解释。不同tokenizer/推理策略也不宜直接按token量比较推理质量。
6. 是否推理正确但底层失败？Vision Oracle有可执行程序、抓取抬升成功而放置失败。模型组没有足够证据将失败归为底层控制；DeepSeek Seed2的主要问题是输出被截断。
7. 视觉是否显著改变模型表现？**无法回答**：Gate B未通过，两模型Vision组都没运行。五场景Pilot也不作统计显著性结论。

逐API物体估计/真值误差、目标TCP/FK残差、抬升、释放速度/角速度和接触见 run_02/analysis/ 每Trial文件；最终成功始终来自未改动的独立物理评分器，非代码文本或sandbox返回值。

## 8. 请求、Token 与费用

| Trial | HTTP | Header request ID | Response ID | finish_reason | input/output/reasoning | 费用估算 USD |
|---|---:|---|---|---|---:|---:|
| deepseek-privileged-seed-0 | 200 | 未返回 header ID | bfaa678d-b22e-495a-a988-c8daa53da92e | stop | 1059 / 3463 / 3219 | 0.00706570 |
| deepseek-privileged-seed-1 | 200 | 未返回 header ID | abec5360-85ac-4275-9459-315f154cf4ad | length | 1059 / 4096 / 4096 | 0.00831904 |
| deepseek-privileged-seed-2 | 200 | 未返回 header ID | 8ddba224-2b9d-4e69-90da-5beaed8da53f | length | 1057 / 4096 / 3776 | 0.00831772 |
| deepseek-privileged-seed-21 | 200 | 未返回 header ID | a9b509d3-c262-4a5b-808c-3c69c3f4af51 | length | 1058 / 4096 / 4096 | 0.00831838 |
| deepseek-privileged-seed-3 | 200 | 未返回 header ID | 46f4ac08-af51-46d8-97dc-94071aa5ff53 | length | 1057 / 4096 / 4096 | 0.00831772 |
| openai-privileged-seed-0 | 200 | req_311ab9c050ea465fbb4d66289c82eeda | chatcmpl-Ej51bTyFeM6nsEqdlfDLYQqeT7Sdq | stop | 951 / 423 / 81 | 0.03066000 |
| openai-privileged-seed-1 | 200 | req_e8ef371fa683431996f90fe709ba72ed | chatcmpl-Ej53ss7SVI1uPZo0pEkhTKYA9Firu | stop | 951 / 373 / 61 | 0.02816000 |
| openai-privileged-seed-2 | 200 | req_3ddd85cd257e48f5b8c0cd9b1f906ba3 | chatcmpl-Ej545ivLVffAX0IjB0BbqxmEsOSev | stop | 949 / 403 / 79 | 0.02964000 |
| openai-privileged-seed-21 | 200 | req_b1245cb729624e0aa40b528bf355b8c2 | chatcmpl-Ej51pzGsvzz6WS1D6k4xorRLUz8EY | stop | 950 / 378 / 70 | 0.02840000 |
| openai-privileged-seed-3 | 200 | req_5d11cdffc19d4791a097fab47056db58 | chatcmpl-Ej565M2PM5fXxxydZceSstcg73zTh | stop | 949 / 474 / 140 | 0.03319000 |

合计公开价格估算 **0.19038854 USD**。DeepSeek输入区分cache hit/miss，调用处于off-peak；GPT按标准短上下文输入/缓存/输出计费。reasoning已包含output，不重复计算。
这不是账单回执；准确消费以账户账单为准。预算执行使用更保守的peak全输入账本，所以DeepSeek预算账本高于这里的实际usage估算。没有任何一次重试或未返回usage的模型请求。

[DeepSeek官方价格](https://api-docs.deepseek.com/quick_start/pricing/)
[GPT-6 Astra官方模型与价格](https://developers.openai.com/api/docs/models/gpt-6-astra)
[OpenAI推理参数](https://developers.openai.com/api/docs/guides/reasoning)

## 9. 视频与Trace索引

正式20个Trial的结果、Trace、视频和严格递增时间戳均核验存在。空程序失败也保存了初始场景单帧视频，没有只保留成功视频。各Trial目录保存prompt、API docs、公开观测、原始响应、生成代码、HTTP尝试与最终物理结果。

| Trial | Reward | 失败分类 | 结果 | Trace | 视频 |
|---|---:|---|---|---|---|
| deepseek-privileged-seed-0 | 1 | none | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/deepseek/privileged/seed-0/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/deepseek-privileged-seed-0/7992b5b90c7f45138024471457b5413b/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_deepseek-privileged-seed-0.mp4) |
| deepseek-privileged-seed-1 | 0 | token_budget_exhausted_before_code | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/deepseek/privileged/seed-1/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/deepseek-privileged-seed-1/bccf5da352ed4f02a5974743ace1d97a/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_deepseek-privileged-seed-1.mp4) |
| deepseek-privileged-seed-2 | 0 | token_budget_truncated_program | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/deepseek/privileged/seed-2/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/deepseek-privileged-seed-2/cb1fc871cf8b434f9606346a04eba044/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_deepseek-privileged-seed-2.mp4) |
| deepseek-privileged-seed-21 | 0 | token_budget_exhausted_before_code | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/deepseek/privileged/seed-21/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/deepseek-privileged-seed-21/3a543e6b49fa49f8a06c205a6c82b268/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_deepseek-privileged-seed-21.mp4) |
| deepseek-privileged-seed-3 | 0 | token_budget_exhausted_before_code | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/deepseek/privileged/seed-3/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/deepseek-privileged-seed-3/29002e3498184d3b96f5028c811235db/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_deepseek-privileged-seed-3.mp4) |
| openai-privileged-seed-0 | 1 | none | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/openai/privileged/seed-0/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/openai-privileged-seed-0/aba9aea9dbdd42b7b39f02312c7e3368/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_openai-privileged-seed-0.mp4) |
| openai-privileged-seed-1 | 1 | none | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/openai/privileged/seed-1/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/openai-privileged-seed-1/fa8e3d6bcab842c8af6c9a8bd89d1279/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_openai-privileged-seed-1.mp4) |
| openai-privileged-seed-2 | 1 | none | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/openai/privileged/seed-2/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/openai-privileged-seed-2/9c54a13c37eb422fbbd1ef21d644cc1b/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_openai-privileged-seed-2.mp4) |
| openai-privileged-seed-21 | 1 | none | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/openai/privileged/seed-21/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/openai-privileged-seed-21/6e72754c19bc4636b071394c779391a9/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_openai-privileged-seed-21.mp4) |
| openai-privileged-seed-3 | 1 | none | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/openai/privileged/seed-3/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/openai-privileged-seed-3/6eb0149d095341038db9e7a89cf3278a/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_openai-privileged-seed-3.mp4) |
| oracle-privileged-seed-0 | 1 | none | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/oracle/privileged/seed-0/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/oracle-privileged-seed-0/375e268e2f524099b0005ea9ed6a957b/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_oracle-privileged-seed-0.mp4) |
| oracle-privileged-seed-1 | 1 | none | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/oracle/privileged/seed-1/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/oracle-privileged-seed-1/a5c374a2bfc0469387638bd19f438007/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_oracle-privileged-seed-1.mp4) |
| oracle-privileged-seed-2 | 1 | none | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/oracle/privileged/seed-2/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/oracle-privileged-seed-2/124d4c20fe5949a69f9a9e9a8e6ddd51/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_oracle-privileged-seed-2.mp4) |
| oracle-privileged-seed-21 | 1 | none | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/oracle/privileged/seed-21/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/oracle-privileged-seed-21/93fda14823084745b21903fdf3e6ed1b/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_oracle-privileged-seed-21.mp4) |
| oracle-privileged-seed-3 | 1 | none | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/oracle/privileged/seed-3/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/oracle-privileged-seed-3/b16cff46e17049dbbd67b41957e097d3/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_oracle-privileged-seed-3.mp4) |
| oracle-vision-seed-0 | 1 | none | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/oracle/vision/seed-0/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/oracle-vision-seed-0/10cb8413921a4eb9a70969f79bd24918/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_oracle-vision-seed-0.mp4) |
| oracle-vision-seed-1 | 0 | placement_failure_after_lift | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/oracle/vision/seed-1/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/oracle-vision-seed-1/8b35c746c9c64a2faa64183731999b40/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_oracle-vision-seed-1.mp4) |
| oracle-vision-seed-2 | 1 | none | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/oracle/vision/seed-2/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/oracle-vision-seed-2/4c37c7ed85984e71895b1438e8d4911a/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_oracle-vision-seed-2.mp4) |
| oracle-vision-seed-21 | 0 | placement_failure_after_lift | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/oracle/vision/seed-21/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/oracle-vision-seed-21/7647982a585d4e4ab3b8ac36a5b82483/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_oracle-vision-seed-21.mp4) |
| oracle-vision-seed-3 | 0 | placement_failure_after_lift | [结果](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/oracle/vision/seed-3/result.json) | [Trace](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/traces/oracle-vision-seed-3/a6dd3ba1771b4e3c81f14518bf3362cf/trace.jsonl) | [视频](/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/pilot_2026-11-11/run_02/videos/video_oracle-vision-seed-3.mp4) |

## 10. 异常、停止与保留

首个实验入口版本在enable_video_capture自动录初始帧之后，又手动录了一帧，造成重复时间戳。最初Privileged/Seed0与Vision/Seed0虽物理reward=1，但视频导出失败，标记system_error并停止对应阶段，未发模型请求，未计入正式Gate。
两条原记录保留在顶层oracle/和traces/，顶层manifest/gates.json为失败Gate。**它们没有成功导出视频，原帧当时未持久化，这一证据缺口明确保留。** 没有用后续视频冒充原失败尝试。
新增回归先重现失败，再删除实验入口多余录帧并增加失败帧保存；测试通过后，在全新run_02目录从头完成Oracle。没有删失败样本或改Reward/阈值。

正式run_02没有Snapshot不一致、OOM、Seed回显失败、API鉴权失败、超时或Sandbox系统故障。四个DeepSeek失败为输出预算下的生成/程序完整性失败，保留原请求/响应，不发恢复请求。

## 11. 后续建议

建议仅对 Privileged 进入20场景验证，先固定本轮参数/分母/失败分类并保留反例。若研究“同等最终代码长度”或“显式低推理配置”，应另开预注册实验，不能与当前4096 completion预算结果混合或补跑覆盖。
Vision先修正定位、抓取TCP和放置目标/到位误差，使用新的Oracle场景集验收，达到Gate后再比较模型。不要从当前GPT5/5直接启动大规模Teacher/SFT/GRPO。

## 12. 修改与复现实验入口

修改文件：capx/envs/reproducibility.py（v2静止重放）、capx/envs/tasks/audited_stack.py（reset消费Snapshot）、capx/llm/client.py（有界重试/响应元数据）、scripts/local/run.py（manifest CLI）；新增capx/experiments/__init__.py、pilot.py和tests/pilot/test_contracts.py。
Reward、Sandbox未改。未训练、下载、升级或操作真实机器人。

完整运行配置：run_02/manifest/experiment_config.json；Gate：run_02/manifest/gates.json；环境/源码哈希：run_02/analysis/provenance.json；配对汇总：run_02/analysis/summary.json；保护核验：run_02/analysis/protection.json；GPU前后状态：gpu_logs/。
所有正式结果目录使用独占创建，重复执行会拒绝覆盖。不要将旧的失败Gate或预检prepared-config-v2误当本轮正式输入。

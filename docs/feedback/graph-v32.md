# V32 原生腕部 F/T 与有界附着恢复

正式工程 `/home/hanyu/dqb/cap-x-20261106`，沿用 V31 统一入口、当前环境和服务。此版本只对 Panda / PickPlace 开放力觉对比；不连接真实机器人。BOX 采集板与传感器本体型号均未核实，见 [硬件审计](../reports/v32-hardware-ft-audit-2026-11-24.md)。

## 常用命令

```bash
cd /home/hanyu/dqb/cap-x-20261106

# 无模型请求：解析配置
scripts/local/infer.sh run --task pick_place --methods G0 G1 --force-mode compare --skill v32 --dry-run

# 单次真实 DeepSeek + Vision + F/T
scripts/local/infer.sh run --task pick_place --methods G1 --force-mode on --skill v32 --seeds 0

# 四组共享初始 DeepSeek 程序 / Snapshot；产生 API 费用
scripts/local/infer.sh run --task pick_place --methods G0 G1 --force-mode compare --skill v32 \
  --seeds 0 7 --max-requests 24 --cost-limit-usd 5 --wall-seconds 7200 --max-tokens 32768

# 工作台，原交互推理入口保留
scripts/local/infer.sh serve --host 127.0.0.1 --port 8200
```

`force-mode off / on / compare` 分别为 Vision-only、Vision+F/T、两种观测 × G0/G1。四组名称为 `A-Vision-G0 / B-Vision-G1 / C-FT-G0 / D-FT-G1`。选择 compare 会自动使用 V32 Skill；四组底座、误差阈值及预算相同。Vision-only 分支没有 `wrist_ft` 公开观测，也不调用力觉验证器；独立记录传感器仅供操作者离线对照。组间相同 `snapshot_hash` 与 `initial_code_hash` 必须保留。

默认输出为新的 `outputs/inference/<timestamp>`；显式 `--output` 必须是不存在的目录。`env_configs/inference/local-v31.json` 选择本机已有 IK runtime；也可传 `--ik-runtime <现有runtime.json>`。不要直接启动第二份 GPU 模型。

## 零新增 LLM 重放与标定

```bash
# reference 指向一个包含 initial.json、config.json、G1/snapshot.json 的历史配对目录
scripts/local/infer.sh replay --task pick_place --methods G1 --skill v32 --force-mode on \
  --seeds 0 --reference outputs/model_comparison/graph_v31_20261124_201722/cli-pair/seed-0-repeat-0

source scripts/local/common.sh
.venv/bin/python -B scripts/local/force_sensor_gate.py \
  --config env_configs/graph_feedback/development-v32.json \
  --output "outputs/force-calibration-$(date +%Y%m%d-%H%M%S)"

# RUN 为一次统一入口生成的输出目录；此分析读取独立物理结果，禁止进入在线 Agent
.venv/bin/python -B scripts/local/analyze_force_trials.py \
  --run "$RUN" --output "$RUN/force-analysis"
```

如果本机 IK runtime 对应进程已退出，可在单独终端使用已有 CPU 服务脚本：

```bash
source scripts/local/common.sh
.venv/bin/python -B scripts/local/start_feedback_ik.py \
  --runtime-file "$NEW_IK_RUNTIME" --warmup-requests "$EXISTING_IK_REQUESTS"
```

两个变量分别为新的 runtime 文件与已有包含两种 IK 请求的审计 JSONL；脚本自动选择空闲端口，不改既有文件。不要用旧 runtime 文件覆盖新服务地址。

## 传感器与安全边界

- 使用 Panda 原模型 `force_ee / torque_ee`，不从物体接触列表合成 Agent 的力信号。只读取原生传感器、机器人编码器、FK 和机器人自身惯性参数。
- 实验发现 step 返回的 sensordata 与当前状态不同步：在独立 `MjData` 中 copy + forward 更新传感器，原仿真状态 / 控制 / RNG 不变。
- MuJoCo 内部反作用力取负后减工具重力；输出基座坐标系、TCP 原点及 TCP 坐标系两个 wrench。平移原点时包含力矩的叉乘项。
- 控制周期约 20 Hz；并非高频真实传感器驱动。记录 sim timestamp、sequence、有效期、原始值、补偿值、滤波值、TCP、夹爪开度与编码器速度。
- 80 ms 一阶滤波供显示，过载保护使用未滤波值，避免隐藏峰值。默认无噪声和零偏置，是理想仿真配置，不是硬件标定结果；配置支持独立噪声 / 偏置。
- 六轴已知小载荷实测支持开发误差包络 5 mN / 2 mNm。短窗口承载判断依赖误差包络，不按物体名称设置质量。未实现完整动力学补偿；运动中的值不能当作准静态质量估计。
- 已确认持物的公开承载估计可用于接触候选的负载重力扣除；原始力记录仍保留负载。未知物体质心 / 载荷力矩不会被假定为已知。
- 40 N / 4 Nm 是本仿真的保守边界停止值，不是硬件额定值。每个控制步检查，超出后固定当前关节目标并中断 Skill；不宣称具有积分子步内取消或工业阻抗控制。

## 恢复与共同能力

`detect_contact`、`verify_grasp_force`、`monitor_slip`、`force_guarded_motion` 由父进程 Skill 使用；`recover_attachment` 复用现有 `confirm_attachment` 契约，并未开放任意原始动作绕过 Sandbox。

力觉承载窗口可支持额外有限等待，最多 20 × 0.25 秒。夹爪必须仍满足原 0.5 mm 收敛要求；无载荷、无进展、传感器失效或预算耗尽时退出。随后仍必须执行已有安全走廊内 6–20 mm 共动验证，满足原 4 mm / 3° 相对运动阈值及当前视觉检查。

V32 共同底座允许用当前可见、非退化点云作为跨运动配准参考，避免把合成矩形表面当作圆润物体的真实形状。原几何边界、形状记忆、碰撞筛选和释放阈值不变；新模板只是待验证假设，共动失败会恢复旧假设。该能力对四组相同，不是 G1 专有策略。

力觉不单独证明力闭合、物体身份、无滑移或安全释放。图沿用通用条件回归与局部修复，正常/异常分支不含任务颜色、Seed 或 Bread 专用动作。

## 记录与前端

每个 episode 保存 `force.jsonl`、`force-calibration.json`、`attachment-evidence/*.npz`、现有 trace / 请求 / 程序 / 实际图 / 视频。`force.jsonl` 是完整采样记录；图索引短窗口指标和来源 ID，不把全时序塞入模型 Prompt。

工作台 `/#experiments` 支持力觉配置、历史六轴曲线、夹爪与 TCP 曲线、时间轴播放/暂停、局部时间窗口和关联恢复事件。Vision-only 的曲线明确标为离线诊断。视频自身仍使用原有 frame→sim_time sidecar；视频播放秒数与仿真秒数不能直接等同。

结果与能力限制见 [V32 开发报告](../reports/graph-feedback-v32-development-2026-11-24.md)。未完成四物体任务的 episode 一律保持物理失败，不因恢复成功或完成部分目标改成成功。

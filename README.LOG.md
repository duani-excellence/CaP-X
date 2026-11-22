# Quick Start — 安全日志分析工具

机器人安全日志分类工具，在浏览器中上传日志即可得到异常事件统计与处理建议。

## 1. 登录云服务器(根据实际情况填写)

```bash
ssh -p 25446 root@connect.nmb2.seetacloud.com
```

密码：`<SSH 密码>`

## 2. 启动 / 停止 / 状态

```bash
safelog-start
```

```bash
safelog-stop
```

```bash
safelog-status
```

- 秒级启动，看到“启动成功”即可。
- 服务在 tmux 后台运行（会话名 `safelog`），关闭 SSH 不影响。
- 与 AICoder（`coder-start`，端口 6006）互不影响，可同时运行。
- 不带参数时读取 `scripts/cloud/safelog.env` 中的配置。

## 3. 访问界面

AutoDL 不支持公网访问，需通过 SSH 隧道：

1. 在**自己电脑上另开一个终端窗口**，执行：

   ```bash
   ssh -N -L 6008:127.0.0.1:6008 -p 25446 root@connect.nmb2.seetacloud.com
   ```

   同时使用 AICoder 时，一条隧道转发两个端口：

   ```bash
   ssh -N -L 6006:127.0.0.1:6006 -L 6008:127.0.0.1:6008 -p 25446 root@connect.nmb2.seetacloud.com
   ```

2. 提示输入密码时，输入：`<SSH 密码>`（输入时不显示字符，输完回车即可）
3. 回车后**没有任何输出、光标停住是正常的**，说明隧道已建立；这个窗口保持打开，关闭即断开访问。
4. 浏览器打开 http://localhost:6008

## 4. 使用

1. 将 `.log` 文件拖入页面（或点击“选择日志”），也可点击“分析内置示例”。
2. 页面展示日志行数、时间跨度、有记录的小时数、异常事件列表（时间、次数、涉及关节、证据行、处理建议）。
3. 当前识别两类异常：

   | 类别 | 级别 | 匹配的错误码 |
   |---|---|---|
   | FSoE 安全通信/参数异常 | critical | `eFSoEInvalidApplicationParams`、`eErrorFsoeStackRecoverableUnkCmd(1204)`、`eErrorFsoeStackRecoverableInvUsrPara(1213)` |
   | 关节扭矩过载 | high | `eErrorSharedProcDataTorqueOverload(205)` |

4. 分析规则：
   - 同类报警相邻间隔不超过 30 秒，合并为一个事件；
   - 无时间戳的续行（以 Tab 开头的多行记录）归属上一条带时间戳的记录；
   - 不带时区的时间戳按 UTC 处理。
5. 上传内容只在内存中流式分析，不落盘；单次上传上限 2 GiB。

验收建议：`active_hours >= 100` 且 `categories_found == 2`。首尾时间跨度超过 100 小时不等于实际采集了 100 小时。

## 5. 可选参数（`safelog-start -h` 查看全部）

| 参数 | 说明 | 示例 | 默认 |
|---|---|---|---|
| `-p` | Web 端口（6006 已被 AICoder 占用） | `-p 6008` | `6008` |
| `-s` | 页面内置示例日志 | `-s "/root/autodl-tmp/logs/robot-001.log"` | `data/safety-log.log` |
| `--max-upload` | 上传上限（字节） | `--max-upload 4294967296` | `2147483648` |
| `--save` | 把本次参数写入 safelog.env，下次直接 `safelog-start` | `--save` | |

修改参数需先 `safelog-stop` 再 `safelog-start` 才会生效。

## 6. 日志与目录

```bash
tmux attach -t safelog                                         # 实时输出；退出按 Ctrl-b 再按 d
tail -f /root/autodl-tmp/safety-log-analyzer/logs/latest.log  # 访问日志
```

- 代码：`/root/autodl-tmp/safety-log-analyzer`（`analyzer.py` 规则与解析，`app.py` HTTP 服务，`static/index.html` 页面）
- Python 环境：项目自带 uv 虚拟环境 `.venv`（CPython 3.12，仅标准库，与 AICoder 环境相互独立）
- 单元测试：`cd /root/autodl-tmp/safety-log-analyzer && .venv/bin/python -m unittest discover -s tests -v`

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| 端口被占用 | `safelog-stop`；或换端口 `safelog-start -p <端口>` |
| 启动失败 | `tail -100 /root/autodl-tmp/safety-log-analyzer/logs/latest.log` |
| 实例重启后 | 数据都在，直接 `safelog-start` |
| `.venv` 损坏 | `bash /root/autodl-tmp/safety-log-analyzer/scripts/cloud/setup_env.sh` |

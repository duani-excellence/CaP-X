# Quick Start

## 1. 登录云服务器 (根据实际情况填写)

```bash
ssh -p 25446 root@connect.nmb2.seetacloud.com
```

密码：`<SSH 密码>`

## 2. 启动 / 停止 / 状态

```bash
coder-start
```

```bash
coder-stop
```

```bash
coder-status
```

- 首次启动加载模型约 1-2 分钟，看到“启动成功”即可。
- 服务在 tmux 后台运行，关闭 SSH 不影响。
- 不带参数时读取 `scripts/cloud/cloud.env` 中的配置。

## 3. 访问界面

AutoDL 不支持公网访问，需通过 SSH 隧道：

1. 在**自己电脑上另开一个终端窗口**，执行：

   ```bash
   ssh -N -L 6006:127.0.0.1:6006 -p 25446 root@connect.nmb2.seetacloud.com
   ```

2. 提示输入密码时，输入：`<SSH 密码>`（输入时不显示字符，输完回车即可）
3. 回车后**没有任何输出、光标停住是正常的**，说明隧道已建立；这个窗口保持打开，关闭即断开访问。
4. 浏览器打开 http://localhost:6006

## 4. 可选参数（`coder-start -h` 查看全部）

| 参数 | 说明 | 示例 | 默认 |
|---|---|---|---|
| `-k` | API key | `-k "sk-xxxx"` | 个人 API key（已写在 cloud.env 的 `CAPX_API_KEY`） |
| `-m` | 模型：`deepseek-v4-pro` / `deepseek-flash` | `-m "deepseek-flash"` | `deepseek-v4-pro` |
| `-p` | Web UI 端口（6008 已被安全日志工具占用） | `-p 6006` | `6006` |
| `-c` | 任务配置 yaml | `-c "env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml"` | `env_configs/cube_stack/franka_robosuite_cube_stack.yaml` |
| `-u` | OpenAI 兼容接口地址 | `-u "https://api.deepseek.com/v1/chat/completions"` | `https://api.deepseek.com/v1/chat/completions` |
| `--save` | 把本次参数写入 cloud.env，下次直接 `coder-start` | `--save` | |

组合示例：

```bash
coder-start -k "sk-xxxx" -m "deepseek-flash"
```

修改参数需先 `coder-stop` 再 `coder-start` 才会生效。

## 5. 日志与结果

```bash
tmux attach -t coder                             # 实时输出；退出按 Ctrl-b 再按 d（不要按 Ctrl-C）
tail -f /root/autodl-tmp/capx/logs/latest.log    # 运行日志
ls /root/autodl-tmp/capx/outputs/                 # 每次任务的代码、视频、执行记录
```

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| 端口被占用 / 残留进程 | `coder-stop`，仍不行用 `coder-stop --force` |
| 启动失败 | `tail -100 /root/autodl-tmp/capx/logs/latest.log` 查看报错 |
| 实例重启后 | 数据都在，直接 `coder-start` |

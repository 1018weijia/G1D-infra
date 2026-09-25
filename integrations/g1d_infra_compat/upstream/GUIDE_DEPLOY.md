# Policy 部署、回退与对齐接管

`policy_deploy.py` 在同一条 30 Hz 主循环里切换控制权：云端 policy 与 XR 遥操作不会同时发关节命令。

采数请用 `collect.py`。本程序的回退和接管只服务于 policy rollout。

## 状态机

```text
startup
    --> READY_POSE --> POLICY_IDLE   # 先抬到准备姿势并保持，此时还不推理
POLICY_IDLE
    --S--> POLICY_LIVE               # 从已经到位的准备姿势开始推理
POLICY_LIVE
    --keyboard B--> POLICY_ROLLBACK --> ALIGNING
ALIGNING
    --keyboard S--> POLICY_LIVE     # 跳过接管，从当前姿态恢复 Policy
    --gamepad A--> TELEOP_LIVE      # 相对接管；前 alignment_handoff_seconds 做 blend
TELEOP_LIVE
    --gamepad A 或 keyboard S--> POLICY_LIVE
    --keyboard B--> POLICY_ROLLBACK # 含 blend 期间
```

回退播完后 **hold 最后一拍命令**（关节、力矩、夹爪），不要改成实测关节角，否则机械臂会从回退终点抽回去。该命令的腰系末端位姿经 FK 变成 OpenXR TARGET，供对齐使用。

## 按键

回退在 **ssh 终端键盘**上，接管和交回在 **右手柄 A** 上。按住手柄 A 只算一次，松开再按才是下一次。

| 键 | 有效阶段 | 作用 |
|---|---|---|
| 键盘 `S` | `POLICY_IDLE` | 从已经到位的准备姿势开始 Policy 推理 |
| 键盘 `S` | `ALIGNING` | 跳过接管，从当前姿态恢复 Policy |
| 键盘 `S` | `TELEOP_LIVE` | 从当前遥操作姿态交回 Policy |
| 键盘 `B` | `POLICY_LIVE`、`TELEOP_LIVE` | 停 Policy / 遥操作，倒放回退缓冲 |
| 手柄 `A` | `ALIGNING` | 相对接管 |
| 手柄 `A` | `TELEOP_LIVE` | 交回 Policy；必须先松开，再过防抖 |
| 键盘 `Q` | 任意 | 退出 |

终端里的 `A` 仍等效于手柄 A，方便没有头显时排查。正常操作用右手柄 A。

第一次手柄 A 后约 0.5 秒内的再按会被忽略。交回必须是：**松开 A，等到 `Teleoperation handoff active`，再按一次手柄 A**。

回退倒放走到最后约 0.5 秒的路径时会放慢，用大约 1.5 秒减速到终点再停住，避免末端突然停住引起抖动。停住后仍然 hold 最后一拍命令，不要改成实测关节角。

## 启动

先在 GPU 机上把推理服务跑起来，再在机器人上：

```bash
cd ~/g1d_infra
SSH_HOST=... SSH_PORT=... SSH_USER=... SSH_KEY=... \
REMOTE_POLICY_HOST=... REMOTE_POLICY_PORT=5555 \
INSTRUCTION="your task" \
./scripts/run_deploy.sh
```

只检查配置、不连云端或机器人：

```bash
./scripts/run_deploy.sh --dry-run
```

脚本会建立 `127.0.0.1:15555` 到远端推理端口的 SSH 隧道，然后启动 `policy_deploy.py`。退出时关掉本脚本创建的隧道。程序起来后会先用 smoothstep 在默认 3 秒内把双臂抬到 `configs/ready_pose.json` 的准备姿势并停住。看到 `Ready pose reached and held` 之后，再按键盘 `S` 才采集第一帧观测并请求动作。抬手过程中按的 `S` 不算，需要到位后再按一次。抬手时按 `Q` 会中止并退出。

常用环境变量：`SSH_HOST`、`SSH_PORT`、`SSH_KEY`、`REMOTE_POLICY_HOST`、`REMOTE_POLICY_PORT`、`IMAGE_HOST`、`UNITREE_DDSINTERFACE`、`INSTRUCTION`、`CONFIG_PATH`、`ROLLBACK_SECONDS`、`READY_POSE_CONFIG`、`READY_POSE_SECONDS`、`TUNNEL_WAIT_SECONDS`（默认 60，跳板机慢时再加大）。

准备姿势默认取自这台 G1-D 已验证的遥操作启动关节命令，顺序是左臂 7 关节、右臂 7 关节。若现场要校准姿势，复制并修改 `configs/ready_pose.json`，再通过 `READY_POSE_CONFIG` 指向新文件；不要把过渡时间设为 0，程序会拒绝瞬间跳到目标。

可选 `--record` 把部署过程存成 episode（默认关）。

## 操作闭环

1. 打开 Vuer 网页并进入 XR，手柄 tracking 有效。启动后机械臂会自己抬到准备姿势并停住，日志出现 `Ready pose reached and held`。
2. 终端按 `S`，从该准备姿势开始 Policy。按 `S` 之前不会请求推理。
3. 需要接管时在终端按 `B`，等待回退完成。机械臂应减速停在回退终点，末端不要抖。
4. 将手柄 RGB 坐标轴与画面 / XR 里的 TARGET 对齐（默认位置 ≤ 4 cm，旋转 ≤ 0.20 rad，稳定 0.5 s）。
5. 按右手柄 `A` 开始相对遥操作。blend 期间增益从 0 升到 1。
6. 松开手柄 A。看到 `Teleoperation handoff active` 后再按一次手柄 `A`（或终端 `S`），从当前姿态交回 Policy。
7. 可重复 `键盘 B → 对齐 → 手柄 A → 手柄 A`。

对齐目标位姿在 `configs/alignment_targets.json`，可用 `--alignment-target-config` 覆盖。`--alignment-forward-offset` 把 TARGET 沿头显前向挪一点，方便对轴。

## 注意

- 不要同时运行 `collect.py` 或其他机械臂控制程序。
- 本地 `15555` 端口开着只说明隧道在；远端推理进程必须已经在听。
- tracking 无效时机械臂保持回退终点，第一次 `A` 不会启动接管；第二次 `A` 交回不要求重新对齐。

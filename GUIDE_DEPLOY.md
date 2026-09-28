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
    --gamepad A--> TELEOP_HOLD      # 停住当前关节和夹爪命令，不发观测
    --keyboard B--> POLICY_ROLLBACK # 含 blend 期间
TELEOP_HOLD
    --keyboard S--> POLICY_LIVE     # 从 hold 姿态发观测并恢复 Policy
    --keyboard B--> POLICY_ROLLBACK
```

回退播完后 **hold 最后一拍命令**（关节、力矩、夹爪），不要改成实测关节角，否则机械臂会从回退终点抽回去。该命令的腰系末端位姿经 FK 变成 OpenXR TARGET，供对齐使用。

## 按键

回退在 **ssh 终端键盘**上，接管和交回在 **右手柄 A** 上。按住手柄 A 只算一次，松开再按才是下一次。

| 键 | 有效阶段 | 作用 |
|---|---|---|
| 键盘 `S` | `POLICY_IDLE` | 从已经到位的准备姿势开始 Policy 推理 |
| 键盘 `S` | `ALIGNING` | 跳过接管，从当前姿态恢复 Policy |
| 键盘 `S` | `TELEOP_HOLD` | 从 hold 姿态发观测并恢复 Policy |
| 键盘 `B` | `POLICY_LIVE`、`TELEOP_LIVE`、`TELEOP_HOLD` | 停 Policy / 遥操作 / hold，倒放回退缓冲 |
| 手柄 `A` | `ALIGNING` | 相对接管 |
| 手柄 `A` | `TELEOP_LIVE` | 停住当前关节和夹爪命令，不发观测。须先松开，看到 `Teleoperation handoff active` |
| 键盘 `F` / 左柄 `X` | 正在 `--record` | 关闭当前 episode 并标记 FAILED |
| 键盘 `Q` | 任意 | 退出 |

终端 `A` 等同手柄 A。0.5 秒内的连按无效。第二次 A 只锁住当前命令，夹爪不松开、不跟手；终端 `S` 才发观测。

回退走到最后约 0.5 秒会放慢，约 1.5 秒停在终点。

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

脚本会建立 `127.0.0.1:15555` 到远端推理端口的 SSH 隧道，再跑 `scripts/wait_policy.sh` 做 TCP/ZMQ 探活，通过后才启动 `policy_deploy.py`。退出时关掉本脚本创建的隧道。默认值来自 [`configs/robot_g1d.yaml`](configs/robot_g1d.yaml)，环境变量可覆盖。

程序起来后会先用 smoothstep 在默认 3 秒内把双臂抬到 `configs/ready_pose.json` 的准备姿势并停住。看到 `Ready pose reached and held` 之后，再按键盘 `S` 才采集第一帧观测并请求动作。抬手过程中按的 `S` 不算，需要到位后再按一次。抬手时按 `Q` 会中止并退出。

常用环境变量：`SSH_HOST`、`SSH_PORT`、`SSH_KEY`、`REMOTE_POLICY_HOST`、`REMOTE_POLICY_PORT`、`IMAGE_HOST`、`UNITREE_DDSINTERFACE`、`INSTRUCTION`、`CONFIG_PATH`、`ROLLBACK_SECONDS`、`READY_POSE_CONFIG`、`READY_POSE_SECONDS`、`WAIT_POLICY_TIMEOUT`、`TUNNEL_WAIT_SECONDS`（默认 60，跳板机慢时再加大）、`ROBOT_CONFIG`。

准备姿势默认取自这台 G1-D 已验证的遥操作启动关节命令，顺序是左臂 7 关节、右臂 7 关节。若现场要校准姿势，复制并修改 `configs/ready_pose.json`，再通过 `READY_POSE_CONFIG` 指向新文件；不要把过渡时间设为 0，程序会拒绝瞬间跳到目标。

可选 `--record` 把部署过程存成 episode（默认关）。录制时状态转换会追加写 `intervention.jsonl`（`rollback_start` / `align_ok` / `teleop_enter` / `teleop_hold` / `policy_resume`）；逐帧 `actions.phase` 也会标 `TELEOP_HOLD`。失败用键盘 `F` 或左柄 `X` 标记，与数采语义一致。

## 操作闭环

1. 打开 Vuer 网页并进入 XR，手柄 tracking 有效。启动后机械臂会自己抬到准备姿势并停住，日志出现 `Ready pose reached and held`。
2. 终端按 `S`，从该准备姿势开始 Policy。按 `S` 之前不会请求推理。
3. 需要接管时在终端按 `B`，等待回退完成。机械臂应减速停在回退终点，末端不要抖。
4. 将手柄 RGB 坐标轴与画面 / XR 里的 TARGET 对齐（默认位置 ≤ 4 cm，旋转 ≤ 0.20 rad，稳定 0.5 s）。
5. 按右手柄 `A` 开始相对遥操作。blend 期间增益从 0 升到 1。
6. 松开 A，看到 `Teleoperation handoff active` 后再按一次 A。手臂和夹爪停在当前命令，不松开。
7. 终端按 `S`，从该姿态发观测并恢复 Policy。
8. 可重复 `键盘 B → 对齐 → 手柄 A → 手柄 A → 键盘 S`。

对齐目标位姿在 `configs/alignment_targets.json`，可用 `--alignment-target-config` 覆盖。现场标定：

```bash
python scripts/align_calibrate.py --yes
# 或先看结果不写盘：
python scripts/align_calibrate.py --dry-run
```

`--alignment-forward-offset` 把 TARGET 沿头显前向挪一点，方便对轴。

## 注意

- 不要同时运行 `collect.py` 或其他机械臂控制程序（`/tmp/g1d_arm_owner.lock` 互斥；调试可用 `--force`）。
- 本地 `15555` 端口开着只说明隧道在；`wait_policy` 失败时不要强行上臂，先确认远端推理进程已在听。
- tracking 无效时机械臂保持回退终点，第一次 `A` 不会启动接管；第二次 `A` 交回不要求重新对齐。

# 轨迹重放

`teleop.replay`（`python -m teleop.replay`）把一条录好的 episode 在机器人上**开环**重放：给它 `data.json`，双臂和夹爪就按录制时的关节命令再走一遍。带 `--use-waist` 录的 episode，转腰和弯腰也一起放。不需要 XR、不需要相机、不需要推理服务。

用来验证「采到的数据是不是真的能驱动机器人」、复现某条轨迹的现场、或者在换硬件后做回归。

> **开环**意味着机器人只按记录的关节角走，不看桌面上实际有什么。清空工作区，手放急停上。

## 启动

```bash
cd ~/g1d_infra
./scripts/run_replay.sh ~/unitree_eai_environment/data/pick_place/episode_0000
```

给目录或者目录里的 `data.json` 都行。等价于：

```bash
UNITREE_DDSINTERFACE=eth0 python -m teleop.replay --data-json ~/unitree_eai_environment/data/pick_place/episode_0000/data.json
```

**先 `--dry-run`**。它只加载、校验、打印摘要，完全不碰机器人：

```bash
./scripts/run_replay.sh <episode> --dry-run
```

```text
episode : /home/unitree/unitree_eai_environment/data/pick_place_100/episode_0001/data.json
frames  : 2074 from source index 0..2073  (block: actions)
playback: 30.0 Hz, 69.1 s
left arm : [-1.246, +1.614] rad, max step 0.057
right arm: [-0.644, +1.196] rad, max step 0.055
grippers : left [0.40, 5.40], right [1.06, 5.40]
waist    : not in this episode
speed   : x1 -> 30.0 Hz, 69.1 s per pass, 1 pass(es)
checks  : ok
```

带腰的 episode 会换成下面两行：

```text
waist yaw : [-0.400, +0.400] rad, max step 0.020
waist pitch: [+0.000, +1.200] rad, max step 0.015
```

## 执行流程

1. 加载 `data.json`，取每帧的 `actions.left_arm.qpos`（7）、`actions.right_arm.qpos`（7）、`actions.left_ee.qpos`、`actions.right_ee.qpos`。若有 `actions.waist.qpos`（转腰，1）和 `actions.waist_pitch.qpos`（弯腰，1），一并取走。两列都没有的旧轨迹只放双臂和夹爪。
2. 校验整条轨迹（见下）。不过就直接退出，机器人不动。
3. 连 DDS，先建 IK 再建臂控制器，然后立刻把双臂和转腰写成实测角度。控制器出生时目标全是 0，晚发一步 250 Hz 线程就会把转腰往 0 拧。之后 hold 当前姿态 1 秒。
4. 打印摘要，包括**当前姿态到准备姿势**、**准备姿势到第一帧**的最大关节差；录了腰时还打印当前转腰、弯腰到第一帧的差。然后等你确认。
5. **准备段**：用 smoothstep 在默认 3 秒内移动到 `configs/ready_pose.json` 的双手抬起姿势，并稳定 0.5 秒。这段只动双臂，腰保持刚才读到的角度。
6. **接近段**：再从准备姿势平滑过渡到第一帧，默认 3 秒。录了腰的话，转腰和弯腰也在这段里平滑到第一帧。
7. **重放段**：按录制帧率逐帧下发。转腰走 `rt/lowcmd` 的第 15 个关节，弯腰走 `rt/waistpitchcmd`。
8. 结束后 hold 最后一帧 1 秒。然后弯腰先回零并重新锁上，双臂和转腰再回 home（`--no-go-home` 可关，腰就停在最后一帧）。

弯腰通道的交接顺序不能反，否则中间会有一段谁都不出力、躯干是软的：**开始**时先起 `rt/waistpitchcmd` 发布者、确认在发，再让 `rt/lowcmd` 松手；**结束**时先在 `rt/waistpitchcmd` 上回零，再把实测角度锁回 `rt/lowcmd`，加锁放 `finally` 里保证 `close()` 抛异常也锁。

重放过程中在终端按 **`Q`** 或 **Ctrl+C** 随时中止：先短暂 hold，再回 home 放下双臂（与正常结束相同；`--no-go-home` 可关）。

## 校验

不通过就不会动，退出码 2。五项：

| 检查 | 阈值 | 含义 |
|---|---|---|
| 关节范围 | \|q\| ≤ 3.5 rad | 和 policy 动作 chunk 用的是同一套限制 |
| 夹爪范围 | [-0.1, 5.5] | 超出说明数据不是这台机器录的 |
| 转腰范围 | \|yaw\| ≤ 2.5 rad | 和采数时摇杆积分的限位一致 |
| 弯腰范围 | [-0.05, 2.45] rad | 弯腰控制器的行程是 0 到约 2.36 rad，这里留了一点余量 |
| 帧间跳变 | ≤ 0.25 rad | 正常遥操作在 30 Hz 下约 0.06 rad；手臂、转腰、弯腰用同一条线 |

报错会指到具体帧：

```text
UNSAFE  : arm joint 0 jumps 1.210 rad between frames 14 and 15, over the 0.25 rad limit
refusing to replay this episode. Narrow the range with --start/--end, or raise --max-joint-step if the jump is real.
```

## 常用参数

| 参数 | 作用 |
|---|---|
| `--dry-run` | 只校验和打印，不连机器人 |
| `--speed 0.5` | 半速重放。第一次跑陌生轨迹建议用 |
| `--start N --end N` | 只放一段。跳过坏帧或只看某个动作 |
| `--stride N` | 每 N 帧取一帧，帧率同步降低，总时长不变 |
| `--loop N` | 重复 N 遍，每遍之间自动平滑回到起点 |
| `--source states` | 放实测关节角而不是下发命令。命令是复现录制的那个，实测滞后一个跟踪误差 |
| `--approach-seconds S` | 接近段时长，默认 3 秒。起始姿态差得远就调大 |
| `--ready-pose-seconds S` | 从当前姿态抬手到准备姿势的时长，默认 3 秒，必须大于 0 |
| `--ready-pose-config P` | 覆盖共享的 14 关节准备姿势配置 |
| `--max-joint-step R` | 放宽帧间跳变阈值 |
| `--no-go-home` | 结束后保持最后姿态，不回 home |
| `--yes` | 跳过确认提示（脚本里批量跑时用） |

## 范围

- 重放**双臂 + 双夹爪**，末端执行器按 `dex1_internal`，和 `collect.py`、`policy_deploy.py` 一致。
- **录了腰才重放腰。** `waist.qpos` 是转腰，`waist_pitch.qpos` 是弯腰，缺一列就只动有的那一列。两列都没有时腰保持当前角度，回 home 时转腰随双臂回零，弯腰保持锁死。
- **不重放升降柱和底盘。** 它们录的是速度命令（`torso.qvel`、`chassis.qvel`），开环重放会让底盘自己走起来。
- 不重放图像。`colors/` 只在转换数据集时用。

## 注意

- 启动前不要同时跑 `collect.py`、`policy_deploy.py` 或其他机械臂控制程序。
- 第一帧的姿态可能离机器人当前位置很远。确认提示里的 `approach` 那行会告诉你最大关节差，觉得太大就先手动把机器人摆近一点。
- 失败轨迹（带 `FAILED` 标记）照样能重放，摘要里会提示 `label : this episode is marked FAILED`。

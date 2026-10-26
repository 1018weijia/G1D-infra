# G1D Infra

G1-D 上的单人数采、LeRobot 转换上传，以及 policy 部署（回退与坐标轴对齐接管）。

## 入口和功能

| 入口 | 功能 |
|---|---|
| `collect.py` / `scripts/run_collect.sh` | XR 遥操作录制 episode（默认录制，`--no-record` 关闭） |
| `policy_deploy.py` / `scripts/run_deploy.sh` | 云端 policy 推理；键盘 B 回退；对齐后手柄 A 接管，再按 A 停住，键盘 S 恢复 |
| `teleop.replay` / `scripts/run_replay.sh` | 开环重放一条 episode 的双臂、夹爪；录了腰就连转腰和弯腰一起放 |
| `scripts/stop_control.sh` | 检测是否有程序在控制机械臂；有的话询问 y/n，y 则按正确信号结束它并复检；退出时张夹爪并缓释刚度 |

```text
collect.py  ->  episode_XXXX/data.json + colors/
            ->  scripts/episode_summary.py  (可选)
            ->  data_convert/convert.sh  ->  LeRobot v3.0
            ->  data_convert/upload.sh   ->  ModelScope
policy_deploy.py  <-  SSH 隧道 + wait_policy + ZMQ/WebSocket
```

## 目录

| 路径 | 作用 |
|---|---|
| `teleop/` | XR、IK、DDS、相机、对齐、回退、推理客户端 |
| `collect.py` | 单人数采 |
| `policy_deploy.py` | policy 部署 |
| `teleop/replay.py` | 轨迹开环重放 |
| `configs/robot_g1d.yaml` | 现场默认：相机 IP、DDS、task、policy 端口 |
| `configs/infer_g1d.yaml` | 推理配置（路径为占位） |
| `configs/alignment_targets.json` | 对齐目标位姿 |
| `configs/ready_pose.json` | 启动抬手位姿 |
| `data_convert/` | JSON → LeRobot v3.0，以及 ModelScope 上传 |
| `3rd/lerobot/` | LeRobot 源码（git submodule，`8fff0fde`） |
| `assets/g1/` | 运行时 IK 用的 URDF 和 Pinocchio cache |
| `assets/g1_D/` | 机型参考 URDF，运行时不用 |

## 环境

- 采数 / 部署：G1 遥操作环境（`unitree_sdk2py`、DDS、相机服务）。依赖见 `requirements.txt`。
- 转换 / 上传：`~/miniconda3/envs/unitree_lerobot`，可用 `PYTHON_BIN` 覆盖。

```bash
cd ~/g1d_infra
git submodule update --init --recursive
```

## 常用脚本

```bash
./scripts/run_collect.sh
./scripts/run_deploy.sh          # 隧道后自动 wait_policy
./scripts/align_calibrate.py     # 写出 alignment_targets.json
./scripts/episode_summary.py ~/unitree_eai_environment/data/pick_place
./scripts/stop_control.sh        # 查有没有程序在控制机械臂，可一键结束
```

`collect` 与 `deploy` 互斥占用 `/tmp/g1d_arm_owner.lock`。调试用 `--force` 绕过。

`stop_control.sh` 依次看三件事：进程表里有没有控制入口、手臂所有权锁是否被真实持有、
`rt/lowcmd` 等控制话题上有没有数据在发布。探针拿不到 `rt/lowstate` 时报"无法确认"而不是
"无人控制"，避免把坏掉的探针当成安全的机器人。确认要停时，它先读 `/proc/<pid>/status`
的 `SigCgt`，能给 SIGTERM 就不发 SIGINT——`policy_deploy.py` 和 `collect.py` 只捕获
`KeyboardInterrupt`，而它们的清理逻辑会先把手臂归零位，所以默认不用 SIGKILL。
超时不退出默认也不强杀，需要时加 `--force`。

退出时的缓释：`collect.py`、`policy_deploy.py` 和 `teleop/replay.py` 在退出时都会按同一顺序收尾：
`ctrl_dual_arm_go_home()`（replay 还先 hold 最后一帧 1 秒；轨迹含弯腰时先在 `rt/waistpitchcmd` 上把弯腰回零并重新锁住）→ `arm_ctrl.open_grippers()`
用 1.5 秒把两个夹爪张到 `GRIPPER_OPEN` → `arm_ctrl.release_stiffness()` 用 5 秒把手臂各关节的
`kp` 按 smoothstep 从满刚度降到 0（`kd` 保持不动，所以关节仍有阻尼）→ 停止发布 `rt/lowcmd`。
这样手臂是慢慢垂下而不是在 4 ms 内突然失去支撑。

夹爪在退出时一律回到张开位，所以"控制开始前"和"释放控制后"状态一致。启动路径本来就会
`cycle_grippers_then_open()`（张→合→张）停在张开位，此前退出路径完全没管夹爪，它只是冻结在
操作者最后捏到的角度，于是经常出现一只手闭合、另一只半开的停车状态。
`release_stiffness()` 不缩放夹爪的 kp，夹爪始终保有夹持力，不会因刚度释放而松脱。

时长用 `--release-seconds` 和 `--gripper-open-seconds` 调整，传 0 可跳过对应步骤。
腰部各关节不参与缓释。`g1d_infra_calib` 的 `scripts/calib_arm_server.py` 有同样的收尾逻辑
（先回启动姿态，再张夹爪，再缓释）。

## 文档

- [GUIDE_COLLECT.md](GUIDE_COLLECT.md) 单人数采
- [GUIDE_DEPLOY.md](GUIDE_DEPLOY.md) policy 回退与对齐接管
- [GUIDE_REPLAY.md](GUIDE_REPLAY.md) 轨迹开环重放
- [TESTING.md](TESTING.md) 验收测试（L0–L3 无需硬件，L4–L5 上机）

## 兼容扩展

`integrations/g1d_infra_compat/` 可选，不改变默认数采和部署。

它把 Pico 上升沿按键接到现有 JSONL/XR 桥：右 A 开始/暂停/恢复，左 Y 录制，左 X 将当前 episode 标为失败，右 B 退出。另有异步语音播报，并保留回退、对齐接管和部署入口。

```bash
python -m unittest -v tests/test_g1d_infra_compat.py
```

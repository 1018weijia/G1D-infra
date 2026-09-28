# G1D Infra

G1-D 上的单人数采、LeRobot 转换上传、以及 policy 部署（含回退与坐标轴对齐接管）。

仓库根目录：`/home/unitree/g1d_infra`

## 支持矩阵（产线）

| 项目 | 支持 |
|---|---|
| 机型 | G1-D |
| 末端 | Dex1 internal（`--ee dex1_internal`） |
| 输入 | XR controller（默认）或 hand tracking |
| IK 模型 | `assets/g1/g1_body29_hand14.urdf`（非 `assets/g1_D`） |

多末端驱动文件仍留在仓库内，但不再接入 `collect.py` / `policy_deploy.py`。

## 两个入口

| 入口 | 做什么 | 不做什么 |
|---|---|---|
| `collect.py` / `scripts/run_collect.sh` | XR 遥操作录制 episode（默认录制，`--no-record` 关） | 不回退、不接管、不对齐 |
| `policy_deploy.py` / `scripts/run_deploy.sh` | 云端 policy 推理、B 回退最近几秒、手柄坐标轴对齐后 A 接管 | 不负责日常采数 |
| `teleop.replay` / `scripts/run_replay.sh` | 开环重放一条录好的 episode（双臂 + 夹爪） | 不放底盘升降、不放图像 |

数据流：

```text
collect.py  ->  episode_XXXX/data.json + colors/
            ->  scripts/episode_summary.py  (可选 QA)
            ->  data_convert/convert.sh  ->  LeRobot v3.0
            ->  data_convert/upload.sh   ->  ModelScope
policy_deploy.py  <-  SSH 隧道 + wait_policy 探活 + ZMQ/WebSocket
```

## 目录

| 路径 | 作用 |
|---|---|
| `teleop/` | XR、IK、DDS、相机、对齐、回退、推理客户端 |
| `collect.py` | 单人数采（Dex1 internal） |
| `policy_deploy.py` | policy 部署 + 回退 + 对齐接管 |
| `teleop/replay.py` | 录制轨迹开环重放 |
| `configs/robot_g1d.yaml` | 现场默认（相机 IP、DDS、task、policy 端口） |
| `configs/infer_g1d.yaml` | 推理配置（结构完整，路径为占位） |
| `configs/alignment_targets.json` | 对齐目标位姿 |
| `configs/ready_pose.json` | 启动抬手位姿 |
| `data_convert/` | JSON → LeRobot v3.0，以及 ModelScope 上传 |
| `3rd/lerobot/` | 官方 LeRobot 源码（git submodule，固定 `8fff0fde`） |
| `assets/g1/` | **运行时 IK** URDF + Pinocchio cache |
| `assets/g1_D/` | 机型参考 URDF（运行时不用） |

## 环境

- 采数 / 部署：现有 G1 遥操作环境（`unitree_sdk2py`、DDS、相机服务）。Python 依赖见 `requirements.txt`。
- 转换 / 上传：`~/miniconda3/envs/unitree_lerobot`，可用 `PYTHON_BIN` 覆盖。

初始化 submodule：

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
```

`collect` 与 `deploy` 互斥占用 `/tmp/g1d_arm_owner.lock`；调试可用 `--force` 绕过。

## 文档

- [GUIDE_COLLECT.md](GUIDE_COLLECT.md) 单人数采
- [GUIDE_DEPLOY.md](GUIDE_DEPLOY.md) policy 回退与对齐接管
- [GUIDE_REPLAY.md](GUIDE_REPLAY.md) 轨迹开环重放
- [TESTING.md](TESTING.md) 验收测试（L0–L3 无需硬件，L4–L5 上机）

## G1-D deployment compatibility extension

The local sys01 fork includes `integrations/g1d_infra_compat/`. It adapts the
Pico rising-edge controls used by this repository (right A start/pause/resume,
left Y record, left X mark the current episode failed, right B quit) to the
existing JSONL/XR bridge. It also provides an asynchronous optional voice
announcer and retains the rollback/alignment handoff helpers and deploy entry
point used by this repository. The extension is opt-in and does not change the
default collector or policy deploy behavior.

Run its hardware-free checks with:

```bash
python -m unittest -v tests/test_g1d_infra_compat.py
```

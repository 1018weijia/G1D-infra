# G1D Infra

G1-D 上的单人数采、LeRobot 转换上传、以及 policy 部署（含回退与坐标轴对齐接管）。

仓库根目录：`/home/unitree/g1d_infra`

## 两个入口

| 入口 | 做什么 | 不做什么 |
|---|---|---|
| `collect.py` / `scripts/run_collect.sh` | XR 遥操作录制 episode | 不回退、不接管、不对齐 |
| `policy_deploy.py` / `scripts/run_deploy.sh` | 云端 policy 推理、B 回退最近几秒、手柄坐标轴对齐后 A 接管 | 不负责日常采数 |

数据流：

```text
collect.py  ->  episode_XXXX/data.json + colors/
            ->  data_convert/convert.sh  ->  LeRobot v3.0
            ->  data_convert/upload.sh   ->  ModelScope
policy_deploy.py  <-  远端推理服务（SSH 隧道 + ZMQ/WebSocket）
```

## 目录

| 路径 | 作用 |
|---|---|
| `teleop/` | XR、IK、DDS、相机、对齐、回退、推理客户端 |
| `collect.py` | 单人数采 |
| `policy_deploy.py` | policy 部署 + 回退 + 对齐接管 |
| `configs/infer_g1d.yaml` | 推理配置（结构完整，路径为占位） |
| `configs/alignment_targets.json` | 对齐目标位姿 |
| `data_convert/` | JSON → LeRobot v3.0，以及 ModelScope 上传 |
| `3rd/lerobot/` | 官方 LeRobot 源码（git submodule，固定 `8fff0fde`） |
| `assets/` | URDF 与 meshes |

## 环境

- 采数 / 部署：现有 G1 遥操作环境（`unitree_sdk2py`、DDS、相机服务）。Python 依赖见 `requirements.txt`。
- 转换 / 上传：`~/miniconda3/envs/unitree_lerobot`，可用 `PYTHON_BIN` 覆盖。

初始化 submodule：

```bash
cd ~/g1d_infra
git submodule update --init --recursive
```

## 文档

- [GUIDE_COLLECT.md](GUIDE_COLLECT.md) 单人数采
- [GUIDE_DEPLOY.md](GUIDE_DEPLOY.md) policy 回退与对齐接管

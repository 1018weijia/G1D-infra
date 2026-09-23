# 开启 Stage 2 在线训练

分支是 `feat/rlt-stage2-online`。算法和 `rlt-openpi` 的 `remote-franka` 一致：先收集，回合结束再按 UTD 更新，EXPO 选动作，接管进 intervention buffer，物理回退走 `rewind_exit_correction`，不能回放时走 `rewind_credit_correction`。

和 Franka 不同的只有这些：

- 机器人是 G1D，控制仍是这套 DDS、对齐和 30 Hz 执行。
- 动作是 16 维原始 AbsQpos，`[L7, LG, R7, RG]`，chunk 长度 64。
- 裁剪和残差幅度用倒豆子 checkpoint 里的值（clip 约 `[-2.164, 6.031]`，`edit_scale=0.2`），不改成 Franka 的末端位姿尺度。
- Motus 每次给出一条参考动作。EXPO 的 4 个 base 槽位都是这一条；另外 4 个是 actor 残差，4 个是 BC actor。

不要占用 `15555`。那是另一台 GPU 部署策略的本地口。在线训练走机器人本机 `127.0.0.1:16555`。

## 1. 训练机上启动服务

在 Motus 仓库：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train/serve_rlt_online.py \
  --config configs/rlt_offline_pourbeans.yaml \
  --online-config configs/rlt_online_pourbeans.yaml \
  --actor-checkpoint outputs/rlt_offline_pourbeans/rlt_offline_pourbeans_0922_2129/actor.pt \
  --device cuda:0 \
  --t5-device cpu
```

日志出现 `RLT online server listening on tcp://127.0.0.1:5555` 后再开隧道。服务只绑本机，不要改成 `0.0.0.0`。

前 20 条 chunk（滑窗不算）返回 Motus 参考动作，样本入库，不做梯度。满 20 条之后才走 EXPO actor。成功或失败按下之后，客户端发 `episode_end`，服务才按「本回合 actor 块数 × 5」做更新。

## 2. 从训练机打反向隧道

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:16555:127.0.0.1:5555 unitree
```

机器人上 `ss -ltn | grep 16555` 应看到 `127.0.0.1:16555`。不要让这条隧道去占 `15555`。

## 3. 机器人上开客户端

仓库：`/home/unitree/g1d_infra`，分支：`feat/rlt-stage2-online`。

```bash
cd /home/unitree/g1d_infra
git checkout feat/rlt-stage2-online
git pull --ff-only

SKIP_TUNNEL=1 \
RL_ONLINE=1 \
UNITREE_DDSINTERFACE=eth0 \
IMAGE_HOST=192.168.123.164 \
INSTRUCTION="把方口杯里的红豆，倒进灰色的粗口杯里，倒半杯" \
./scripts/run_deploy.sh
```

`SKIP_TUNNEL=1` 会连 `127.0.0.1:16555`，并强制 `--rl-online`、关闭预取。图像服务和 `teleimager-server` 需要已经在 `192.168.123.164` 上。准备姿势到位后再按键。

## 4. 按键

| 键 | 作用 |
|---|---|
| `S` | 开始一条 episode，发出第一块 `act` |
| `P` | 当前这一拍 +0.5，回合继续 |
| `Y` | 当前这一拍 +1。本块结束后把动作发回，成功收束，然后手臂回到准备姿势。再按 `S` 开下一条 |
| `N` | 本块结束后把已有奖励发回，失败收束，然后手臂回到准备姿势。再按 `S` 开下一条 |
| `A` | 对齐完成后接管。人工关节记成 `intervention=true`。再按 `A` 或 `S` 交回策略 |
| `B` | 物理回退一块。只倒放这一块已经走出的关节指令；当前块还没动时，倒放上一整块。学习侧只把这一块标成坏分支，最后一步奖励 -0.2，切断 bootstrap。没有可回放的历史时发 `rewind_credit`，并给前一段末步 +0.1 |
| `Q` | 退出 |

`Y` / `N` 只在策略正在执行时有效。奖励写在按下的那一拍。动作发回之后手臂回到准备姿势，再按 `S` 开始下一条 episode。

回退之后的下一条真实动作，会和坏分支起点组成偏好对。credit 回退不记偏好对。

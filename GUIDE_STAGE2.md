# 开启 Stage 2 在线训练

分支是 `feat/rlt-stage2-online`。算法和 `rlt-openpi` 的 `remote-franka` 一致：先收集，回合结束再按 UTD 更新，EXPO 选动作，接管进 intervention buffer，物理回退走 `rewind_exit_correction`，不能回放时走 `rewind_credit_correction`。

和 Franka 不同的只有这些：

- 机器人是 G1D，控制仍是这套 DDS、对齐和 30 Hz 执行。
- 动作是 16 维原始 AbsQpos，`[L7, LG, R7, RG]`，chunk 长度 64。
- 裁剪和残差幅度用倒豆子 checkpoint 里的值（clip 约 `[-2.164, 6.031]`，`edit_scale=0.2`），不改成 Franka 的末端位姿尺度。
- Motus 每次给出一条参考动作。EXPO 的 4 个 base 槽位都是这一条；另外 4 个是 actor 残差，4 个是 BC actor。

不要占用 `15555`。那是另一台 GPU 部署策略的本地口。在线训练走机器人本机 `127.0.0.1:16556`。

## 1. 训练机上启动服务

在 Motus 仓库：

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python train/serve_rlt_online.py \
  --config configs/rlt_offline_pourbeans.yaml \
  --online-config configs/rlt_online_pourbeans.yaml \
  --actor-checkpoint outputs/rlt_offline_pourbeans/rlt_offline_pourbeans_0922_2129/actor.pt \
  --device cuda:0 \
  --window-device cuda:1 \
  --t5-device cpu
```

日志出现 `RLT online server listening on tcp://127.0.0.1:5555` 后再开隧道。服务只绑本机，不要改成 `0.0.0.0`。启动要加载两份 Motus，约 4–5 分钟。

`--window-device` 在第二张卡上放一份冻结的 Motus，只在后台编码滑窗帧（每帧约 0.6 s）。`episode_end` 只做更新，几秒内返回，下一条可以马上开始。本回合的滑窗编码完后才入库（日志 `windows added=`），从下一回合的更新开始用上。不加这个参数时，滑窗在 `episode_end` 里同步编码，一条 10 块的轨迹要多等 1–2 分钟。

前 20 条 chunk（滑窗不算）返回 Motus 参考动作，样本入库，不做梯度。满 20 条之后才走 EXPO actor。成功或失败按下之后，客户端发 `episode_end`，服务才按「本回合 actor 块数 × 5」做更新。

## 2. 机器人上开正向隧道

机器人经公网跳板 `123.56.183.38` 直接 ssh 到训练机（`~/.ssh/config` 里的 `motus-gpu`，密钥 `~/.ssh/id_ed25519_gpu`）。在跳板上这把钥匙只能转发到 `127.0.0.1:10099`，也就是训练机的 sshd。放在 tmux `rl-tunnel` 里，断了自动重连：

```bash
tmux new-session -d -s rl-tunnel 'while true; do ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -L 127.0.0.1:16556:127.0.0.1:5555 motus-gpu; sleep 3; done'
```

不要再用训练机经 tailscale 打的 `-R 16555` 反向隧道。那条走 userspace tailscale，实测一帧 act 要 50s，这条是 3s 以内。

图像版式必须和 Stage 1、离线 buffer、Motus 训练一致：头部只用左眼（`cam_left_high`，480x640 缩到 240x320），下面两个腕部各 120x160，上下各 12 px 黑边。头部相机是 480x1280 双目，客户端在拼接前只取左半边；逐像素已和 MotusV2 `_stitch_t_shape` 对齐。旧客户端把整张双目塞进 120 px 的条带，服务端会识别并还原，同时打警告。

图像按 JPEG q90 发送（一帧约 20 KB，原始是 369 KB）。机器人上行只有约 1 Mbit/s，64 步的 transition 从约 24 MB 降到约 1.3 MB。

## 3. 机器人上开客户端

仓库：`/home/unitree/g1d_infra`，分支：`feat/rlt-stage2-online`。

```bash
cd /home/unitree/g1d_infra
git checkout feat/rlt-stage2-online
git pull --ff-only

SKIP_TUNNEL=1 \
RL_LOCAL_PORT=16556 \
RL_ONLINE=1 \
UNITREE_DDSINTERFACE=eth0 \
IMAGE_HOST=192.168.123.164 \
INSTRUCTION="把方口杯里的红豆，倒进灰色的粗口杯里，倒半杯" \
./scripts/run_deploy.sh
```

`SKIP_TUNNEL=1 RL_LOCAL_PORT=16556` 会连 `127.0.0.1:16556`，并强制 `--rl-online`、关闭预取。图像服务和 `teleimager-server` 需要已经在 `192.168.123.164` 上。准备姿势到位后再按键。

## 4. 按键

| 键 | 作用 |
|---|---|
| `S` | 开始一条 episode，发出第一块 `act` |
| `P` | 当前这一拍 +0.5，回合继续 |
| `Y` | 当前这一拍 +1。正在执行的块发回后成功收束。当前没有在执行的块时立刻收束，不再执行下一块。然后手臂回到准备姿势。再按 `S` 开下一条 |
| `N` | 正在执行的块发回后失败收束，保留已有奖励。当前没有在执行的块时立刻收束。然后手臂回到准备姿势。再按 `S` 开下一条 |
| `A` | 对齐完成后接管。人工关节记成 `intervention=true`。接管段在本地按 64 个计入步连续切块，不等服务端，一步不丢；块边界和每 4 步的观测在本地拍下，排队上传，服务端据此在接管段也切滑窗。对齐等待、交接过渡和接管后原地不动的拍都不记；和 Franka 的死区一样，任一只手末端动 1.5 mm、转 0.01 rad、关节变 0.02 rad 或夹爪明显变化才算一步，接管块的观测也在第一次真正动的时候拍。接管中再按手柄 `A`（或终端 `A`/`S`）结束接管并**暂停**：手臂停在当前姿势，没满 64 步的最后一块也照常上传 |
| `B` | 物理回退一块。只倒放这一块已经走出的关节指令；当前块还没动时，倒放上一整块。学习侧只把这一块标成坏分支，最后一步奖励 -0.2，切断 bootstrap。没有可回放的历史时发 `rewind_credit`，并给前一段末步 +0.1 |
| `Q` | 退出 |

`Y` / `N` 只在策略控制时有效。有正在执行的块时，奖励写在按下的那一拍，这块发回之后手臂回到准备姿势。没有正在执行的块时（还在等下一块），立刻结束，不再执行新到的动作，奖励记到最后一块已执行的动作上，然后回到准备姿势。再按 `S` 开始下一条 episode。

接管结束后的暂停（终端按键；手柄 `A` 在暂停时不起作用，防止连按跳过暂停）：

| 键 | 作用 |
|---|---|
| `A` | 继续推理。接管块传完后，策略从当前姿势要下一块 |
| `S` 或 `Y` | 直接成功收束：+1 记在最后一块接管动作上，回到准备姿势 |
| `N` | 失败收束，回到准备姿势 |

暂停期间接管块继续在后台上传。成功/失败的 `episode_end` 排在它们后面，所以总是收束在最后一块人工动作上。

不足 64 步的块（接管结束或按 B 时的最后一块）：只上传真正执行的步数，服务端用最后一个动作补齐到 64 步（原地保持；Franka 用零补齐，但这里是绝对关节角，零是远离当前姿势的无效动作），`executed_steps` 记实际步数；成功奖励写在最后一个执行步；这种块不参与切滑窗。

回退之后的下一条真实动作，会和坏分支起点组成偏好对。credit 回退不记偏好对。

# 验收测试

这份文档用来确认 `g1d_infra` 在一台新机器上是否可用。测试分成六级，**L0–L3 不需要机器人**，任何一台装好环境的机器都能跑；**L4–L5 需要 G1-D 实机、相机服务和 XR 头显**。

先跑完 L0–L3，再上机。L0–L3 全绿说明代码、依赖、数据格式、推理通信都没问题；上机再挂，问题基本在硬件或网络。

| 级别 | 内容 | 需要硬件 | 耗时 |
|---|---|---|---|
| L0 | 环境自检 | 否 | ~30 s |
| L1 | 单元测试 | 否 | ~5 s |
| L2 | 采集格式 → LeRobot 转换 → 上传 → 重放校验（假数据） | 否 | ~60 s |
| L3 | 推理通信链路（mock 推理服务） | 否 | ~10 s |
| L4 | 实机数采 + 轨迹重放 | 是 | ~20 min |
| L5 | 实机 policy 回退与接管 | 是 + GPU 机 | ~30 min |

记录结果用文末的[验收表](#验收记录表)。

---

## 前置准备

```bash
cd ~/g1d_infra
git submodule update --init --recursive
```

两套 Python 环境，各管各的：

| 用途 | 环境 | 说明 |
|---|---|---|
| L0–L1、L3、L4、L5 | 遥操作环境（本机为 `~/miniconda3/envs/tv`） | `unitree_sdk2py`、DDS、pinocchio、vuer |
| L2 的转换与上传 | `~/miniconda3/envs/unitree_lerobot` | `torch`、`datasets`、`av`、`modelscope`；可用 `PYTHON_BIN` 覆盖 |

下文命令里的 `python` 都指**遥操作环境**的 python；`convert.sh` / `upload.sh` 会自己切到 `unitree_lerobot`。

---

## L0 环境自检

检查文件齐不齐、依赖能不能 import、两个入口能不能起来。不碰机器人、不联网。

```bash
cd ~/g1d_infra
python tests/check_env.py
```

**成功标志**

- 最后一行 `all required checks passed`，退出码 0。
- `-- files --`、`-- runtime dependencies --`、`-- repo modules --`、`-- entrypoints --` 四组全是 `[PASS]`。

**允许的 WARN**

- `-- conversion environment --` 出现 `[WARN] python not found: .../unitree_lerobot/bin/python`：这台机器不做数据转换就没关系，L2 会跳过。

**常见失败**

| 现象 | 原因 | 处理 |
|---|---|---|
| `[FAIL] 3rd/lerobot/src/lerobot` | submodule 没拉 | `git submodule update --init --recursive` |
| `[FAIL] unitree_sdk2py` / `vuer` | 用错 conda 环境 | 激活遥操作环境后重跑 |
| `[FAIL] pinocchio`（`cannot allocate memory in static TLS block`） | aarch64 上 `libgomp` 的静态 TLS 问题 | `export LD_PRELOAD=$(find / -name 'libgomp.so.1' 2>/dev/null | head -1)` 后重跑 |

> 脚本刻意让每个 import 各起一个解释器。pinocchio 在 aarch64 上只有**先加载**才稳，混在同一个进程里检查会误报——而实际入口不会踩到。

---

## L1 单元测试

89 个无硬件依赖的用例，覆盖准备姿势、对齐状态机、回退缓冲、接管按键防抖、episode 失败标记与坏轨迹识别、录制写盘吞吐、轨迹重放校验、动作 chunk 校验。

```bash
cd ~/g1d_infra
python -m unittest discover -s tests -v
```

装了 pytest 的话等价于 `python -m pytest -q`（`pytest.ini` 已配好 `pythonpath`）。pytest 在 `requirements.txt` 里，但不装也能用上面的 unittest 跑。

**成功标志**

```text
Ran 89 tests in 10.321s

OK
```

数字可能随版本变，关键是最后一行 `OK`、没有 `FAILED`。

**重点看这几个用例**，它们对应 GUIDE_DEPLOY.md 里最容易出事的逻辑：

| 用例 | 保证什么 |
|---|---|
| `test_rollback_hold_uses_last_command_not_measured_state` | 回退播完后 hold 的是最后一拍命令，不是实测关节角。挂了会导致机械臂从回退终点抽回去 |
| `test_move_commands_target_and_preserves_grippers` | 准备段准确到达抬手姿势，并且不会误开合夹爪 |
| `test_a_takeover_then_held_repeat_does_not_resume` | 按住 A 不会被当成「接管 + 立刻交回」 |
| `test_a_ignored_until_rollback` | 没回退就按 A 不接管 |
| `test_tracking_loss_holds_and_reanchor_restarts_blend` | tracking 掉了保持不动，恢复后重新 blend |
| `test_validate_rejects_bad_chunk` | 越界或含 NaN 的动作 chunk 不下发 |
| `test_success_flag_found_past_the_tail_window` | 真实 `data.json` 很大时，坏轨迹识别仍读得到末尾的 `"success"` |
| `test_create_episode_does_not_build_a_rerun_logger` | 开始录制不会在控制循环里调 `rr.spawn()`，否则机械臂跳变 |
| `test_two_episodes_back_to_back` | 连录两条（一成功一失败）都能保存，第二条不会被「recorder is busy」挡掉 |

---

## L2 数采格式 → 转换 → 上传（假数据）

用 `EpisodeWriter` 生成和 `collect.py` **完全同格式**的假 episode，跑一遍转换和上传预演。这一级验证的是数据链路，不验证遥操作。

### L2.1 生成假 episode

```bash
cd ~/g1d_infra
rm -rf /tmp/g1d_fake
python tests/make_fake_episodes.py \
  --task-dir /tmp/g1d_fake/fake_task \
  --episodes 3 --frames 20 --fail 1
```

**成功标志**

```text
wrote /tmp/g1d_fake/fake_task/episode_0000 success=True
wrote /tmp/g1d_fake/fake_task/episode_0001 success=False
wrote /tmp/g1d_fake/fake_task/episode_0002 success=True
```

目录结构应该是：

```text
/tmp/g1d_fake/fake_task/episode_0000/
  data.json
  colors/000000_color_0.jpg ... 000019_color_3.jpg    # 20 帧 × 4 路相机 = 80 张
  depths/  audios/                                     # 空
/tmp/g1d_fake/fake_task/episode_0001/
  FAILED                                               # 只有失败轨迹有
```

再确认 JSON 顶层字段：

```bash
python -c "
import json; d=json.load(open('/tmp/g1d_fake/fake_task/episode_0001/data.json'))
print(list(d), d['success'], len(d['data']))"
```

应输出 `['info', 'text', 'data', 'success'] False 20`。

### L2.2 转换成 LeRobot v3.0

采集时标失败的轨迹会被自动识别，所以这里**不用**手工指定 `--bad`：

```bash
cd ~/g1d_infra
rm -rf /tmp/g1d_fake/out
./data_convert/convert.sh \
  --raw-dir /tmp/g1d_fake/fake_task \
  --output-dir /tmp/g1d_fake/out
```

中途会刷大量 `libx264` / ffmpeg 日志，属正常。

**成功标志**

```text
Bad   : <none> (auto-detect: on)
Auto-detected failed episodes (FAILED marker or success=false): [1]
...
Wrote LeRobot v3.0 dataset to /tmp/g1d_fake/out
Converted 3 episodes, tagged bad=1, skipped bad=0
Vector features: 17 columns (JSON grouping preserved)
Quality map: /tmp/g1d_fake/out/meta/episode_quality.json
```

输出目录必须长这样：

```text
/tmp/g1d_fake/out/
  data/chunk-000/file-000.parquet
  meta/info.json  meta/stats.json  meta/tasks.parquet  meta/episode_quality.json
  meta/episodes/chunk-000/
  videos/observation.images.cam_left_high/chunk-000/file-000.mp4
  videos/observation.images.cam_right_high/chunk-000/file-000.mp4
  videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4
  videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4
```

关键字段自查：

```bash
python -c "
import json
info = json.load(open('/tmp/g1d_fake/out/meta/info.json'))
q = json.load(open('/tmp/g1d_fake/out/meta/episode_quality.json'))
print(info['codebase_version'], info['robot_type'], info['total_episodes'], info['total_frames'], info['fps'])
print(q['bad_source_episodes'], q['explicit_bad_episodes'], q['auto_detected_bad_episodes'])
print([(e['source_episode'], e['is_bad']) for e in q['episodes']])"
```

期望：

```text
v3.0 Unitree_G1_MoveibleLift_Dex1_NoUseWaist 3 60 30
[1] [] [1]
[(0, False), (1, True), (2, False)]
```

`bad_source_episodes` 是最终生效的并集，`explicit_bad_episodes` 是 `--bad` 给的，`auto_detected_bad_episodes` 是自动识别出来的。这里 episode_0001 只靠自动识别就被标上了。

### L2.2b 坏轨迹来源（三种模式）

坏轨迹判定 = **自动识别** ∪ **手工列表**。自动识别看两个东西：`FAILED` 标记文件，以及 `data.json` 末尾的 `"success": false`。两者任一命中即算坏，`collect.py` 按左 X 时两个都会写。手工列表来自 `--bad` 或 `convert.sh` 顶部的 `BAD_EPISODES`，用于事后才判定为坏的轨迹。

跑这三条确认三种模式都对：

```bash
cd ~/g1d_infra
# 1) 默认：只有自动识别
./data_convert/convert.sh --raw-dir /tmp/g1d_fake/fake_task --output-dir /tmp/g1d_fake/m1
# 2) 自动 + 手工把 episode_0002 也标坏
./data_convert/convert.sh --raw-dir /tmp/g1d_fake/fake_task --output-dir /tmp/g1d_fake/m2 --bad 2
# 3) 关掉自动识别，只信手工列表
./data_convert/convert.sh --raw-dir /tmp/g1d_fake/fake_task --output-dir /tmp/g1d_fake/m3 --no-auto-bad
```

**成功标志**

| 模式 | `Converted ...` | `bad_source_episodes` |
|---|---|---|
| 1 默认 | `tagged bad=1` | `[1]` |
| 2 加 `--bad 2` | `tagged bad=2` | `[1, 2]` |
| 3 `--no-auto-bad` | `tagged bad=0` | `[]` |

模式 3 是旧行为，留着是为了能复现历史数据集；正常采数**不要**用它，否则失败轨迹会被当成好数据。

坏轨迹是**打标签保留**，不是删除。要真的丢掉，加 `--skip-bad`，输出会变成 `Converted 2 episodes, ..., skipped bad=1`。

标签也落到了每一帧，可以这样确认（需要 `unitree_lerobot` 环境）：

```bash
~/miniconda3/envs/unitree_lerobot/bin/python -c "
import pandas as pd
df = pd.read_parquet('/tmp/g1d_fake/out/data/chunk-000/file-000.parquet')
print(df.groupby('complementary_info.source_episode_index')[['next.success','complementary_info.is_bad']].first())"
```

期望 episode 1 那行是 `False / True`，另外两行是 `True / False`。

### L2.3 上传预演

不需要真 token，`--dry-run` 只列文件、不上传。

```bash
cd ~/g1d_infra
./data_convert/upload.sh \
  --token fake-token \
  --repo demo/fake_task \
  --local-dir /tmp/g1d_fake/out \
  --dry-run
```

**成功标志**

```text
Format    : LeRobot dataset (meta/info.json found)
...
Dry run only. No upload.
```

`Files : 10`、`Size` 非零，且列表里同时有 `meta/info.json` 和 4 个 `videos/.../file-000.mp4`。

要测真上传，把 `--token` 换成 [ModelScope access token](https://www.modelscope.cn/my/myaccesstoken)、`--repo` 换成自己的仓库、去掉 `--dry-run`。成功后去网页端确认文件数和上面列出的一致。

### L2.4 重放校验（不碰机器人）

`python -m teleop.replay --dry-run` 只加载、校验、打印摘要，正好可以拿假数据验证重放的解析和安全检查。

```bash
cd ~/g1d_infra
python -m teleop.replay --data-json /tmp/g1d_fake/fake_task/episode_0000 --dry-run
python -m teleop.replay --data-json /tmp/g1d_fake/fake_task/episode_0001 --dry-run | grep label
```

**成功标志**

- `episode_0000`：`label   : success`、`checks  : ok`、结尾 `dry run, robot untouched`。
- `episode_0001`：`label   : this episode is marked FAILED`。
- 两条都是 `frames  : 20`、`playback: 30.0 Hz`。

再确认坏数据会被挡下来：

```bash
python - <<'PY'
import json, numpy as np, os
os.makedirs('/tmp/g1d_fake/jump/episode_0000', exist_ok=True)
blk = lambda l, r: {'left_arm': {'qpos': list(l)}, 'right_arm': {'qpos': list(r)},
                    'left_ee': {'qpos': [1.0]}, 'right_ee': {'qpos': [1.0]}}
data = []
for i in range(30):
    l = np.full(7, i * 0.01)
    if i == 15:
        l = l + 1.2                      # 注入一个 1.2 rad 的跳变
    data.append({'idx': i, 'colors': {}, 'states': blk(l, -l), 'actions': blk(l, -l)})
json.dump({'info': {'image': {'fps': 30.0}}, 'text': {}, 'data': data},
          open('/tmp/g1d_fake/jump/episode_0000/data.json', 'w'))
PY
python -m teleop.replay --data-json /tmp/g1d_fake/jump/episode_0000 --dry-run; echo "exit=$?"
```

**成功标志**：退出码 2，并指到具体帧——

```text
UNSAFE  : arm joint 0 jumps 1.210 rad between frames 14 and 15, over the 0.25 rad limit
refusing to replay this episode. ...
```

这条是重放最重要的护栏：损坏或拼接错的轨迹绝不能下发到机械臂。

### L2.5 清理

```bash
rm -rf /tmp/g1d_fake
```

---

## L3 推理通信链路（mock 推理服务）

`policy_deploy.py` 本身要机器人，但它和云端之间的那段——图像拼接、state 编码、ZMQ 往返、动作 chunk 校验与展开——可以完全离线验证。仓库里带了一个假推理服务，收到什么 state 就原样回一个 64 步的保持动作。

开两个终端。

**终端 A：起假服务**

```bash
cd ~/g1d_infra
python tests/mock_policy_server.py --bind tcp://127.0.0.1:5599
```

**终端 B：跑客户端**

```bash
cd ~/g1d_infra
python tests/check_policy_roundtrip.py --server-port 5599
```

**成功标志**

终端 B：

```text
stitched frame (384, 320, 3) dtype=uint8, state dim 16
action chunk (64, 16), predict 0.2 ms
exec queue 8 steps, first arm_q[:3]=[0. 0. 0.], grippers=(0.000, 0.000)
roundtrip OK
```

终端 A：

```text
served request 1 instruction='pick up the red cup' frame=(384, 320, 3)
```

逐项对应：

- `(384, 320, 3)` 必须和 `configs/infer_g1d.yaml` 里的 `video_height: 384` / `video_width: 320` 一致。对不上说明配置和拼图代码脱节。
- `state dim 16`、`action chunk (64, 16)` 对应 `state_dim: 16` 和 `num_video_frames(8) × video_action_freq_ratio(8) = 64`。
- `exec queue 8 steps` 来自 `--exec-chunk-steps 8`；`action_interp_factor=1` 所以不插值。
- `roundtrip OK` 说明 `validate_action_chunk` 放行了，动作能转成机器人命令。

改 `configs/infer_g1d.yaml` 之后重跑这一级，是确认改动没写错最快的办法。

### L3.1 顺带验证 SSH 隧道（可选）

如果已经有 GPU 机，可以在**不启动机器人**的情况下先验隧道：

```bash
# GPU 机上
python tests/mock_policy_server.py --bind tcp://0.0.0.0:5555

# 机器人上
cd ~/g1d_infra
SSH_HOST=<gpu-ip> SSH_PORT=22 SSH_USER=<user> SSH_KEY=~/.ssh/id_rsa \
REMOTE_POLICY_HOST=127.0.0.1 REMOTE_POLICY_PORT=5555 \
./scripts/run_deploy.sh --dry-run
```

`--dry-run` 只打印配置就退出，**不会**建隧道也不会校验 `INSTRUCTION`：

```text
[launcher] dry-run
[launcher] tunnel: <user>@<gpu-ip>:22 -> 127.0.0.1:5555 via 127.0.0.1:15555
[launcher] protocol: zmq
[launcher] config: /home/unitree/g1d_infra/configs/infer_g1d.yaml
```

想真验隧道，手工拉一条再跑客户端：

```bash
ssh -p 22 -i ~/.ssh/id_rsa -N -L 127.0.0.1:15555:127.0.0.1:5555 <user>@<gpu-ip> &
python tests/check_policy_roundtrip.py --server-port 15555
```

同样看到 `roundtrip OK` 就说明隧道 + 远端服务都通了。

> 本地 `15555` 开着**只**说明隧道在，不代表远端推理进程在听。`run_deploy.sh` 正式启动时会先 `policy_require_remote` 探一下远端端口，探不到会直接报错退出，这是预期行为。

---

## L4 实机数采

到这一步需要：G1-D 本体、`192.168.123.164` 上的相机服务、Pico / OpenXR 头显和手柄。

**先确认没有别的程序在控制机械臂。**

### L4.1 相机服务可达

```bash
cd ~/g1d_infra
python -c "
from teleop.teleimager.src.teleimager.image_client import ImageClient
cfg = ImageClient(host='192.168.123.164', request_bgr=True).get_cam_config()
for k in ('head_camera','left_wrist_camera','right_wrist_camera'):
    print(k, cfg[k]['image_shape'], 'zmq', cfg[k]['enable_zmq'])"
```

**成功标志**：秒回，三行都打印出来，例如

```text
head_camera [480, 1280] zmq True
left_wrist_camera [480, 640] zmq True
right_wrist_camera [480, 640] zmq True
```

卡住或超时 = 相机服务没起来，或 `IMAGE_HOST` 不对。先修这个再往下。

### L4.2 遥操作不录制

先拿 `--no-record` 试手，别一上来就写盘。

```bash
cd ~/g1d_infra
./scripts/run_collect.sh --no-record
```

**成功标志**

1. 终端打印出 Vuer 网页地址，浏览器打开后能进 XR。
2. 头显里能看到机器人视角画面，手柄 tracking 有效。
3. 按**右手柄 A** 后机械臂跟随手柄动，延迟不明显。
4. 再按**右 A** 暂停，机械臂停住；又按一次恢复。
5. 按**右 B** 退出，进程干净结束，机械臂不乱甩。

按键来自手柄，不是 ssh 终端。没开 `--ipc` 时终端上的 `R`/`S`/`F`/`Q` 是备用键。

### L4.3 正式录制

```bash
cd ~/g1d_infra
TASK_DIR=/tmp/g1d_test_data TASK_NAME=smoke_test TASK_GOAL="pick and place" \
./scripts/run_collect.sh
```

按这个顺序走一遍：

| 步骤 | 操作 | 预期 |
|---|---|---|
| 1 | 右 A | 遥操作启动 |
| 2 | 左 Y | 开始录制 `episode_0000` |
| 3 | 动 3–5 秒 | 终端持续刷 item 日志 |
| 4 | 左 Y | 保存并**自动暂停** |
| 5 | 右 A | 恢复，摆下一次场景 |
| 6 | 左 Y | 开始 `episode_0001` |
| 7 | 动几秒后按左 X | 停止并**标记失败** |
| 8 | 右 B | 退出 |

**成功标志**

```bash
find /tmp/g1d_test_data/smoke_test -maxdepth 2 | sort
python -c "
import json
for i in (0,1):
    d = json.load(open(f'/tmp/g1d_test_data/smoke_test/episode_{i:04d}/data.json'))
    print(i, 'success=', d['success'], 'frames=', len(d['data']), 'cams=', list(d['data'][0]['colors']))"
```

- `episode_0000/` 和 `episode_0001/` 都在，各有 `data.json` 和非空 `colors/`。
- `episode_0000` → `success= True`，目录里**没有** `FAILED`。
- `episode_0001` → `success= False`，目录里**有** `FAILED`。
- 两者 `frames` 约等于 `录制秒数 × 30`。
- `cams` 是 `['color_0', 'color_1', 'color_2', 'color_3']`（双目头 + 双腕）。
- `colors/` 里的图片数 = `frames × 相机数`，随便打开几张确认不是全黑。

录制中按右 A 不会暂停，这是设计如此。

**同时盯这几条时序**，它们是首次上机时踩到的两个 bug 的回归点：

| 现象 | 应该是 |
|---|---|
| 步骤 2 按左 Y 到 `New episode created` | 几乎立即。若卡住数秒且机械臂跳一下，说明 Rerun 又被打开了 |
| 步骤 4 按左 Y 到 `Episode saved successfully` | 1 秒内。日志里 `N frames queued` 应该是个小数字 |
| 步骤 6 按左 Y 开始第二条 | 正常创建。若打印 `recorder is busy`，说明写盘线程没跟上 |

录制中每秒会打一行 `==> episode_0000: wrote N frames`，不再是每帧一行。

### L4.4 把实采数据走一遍转换

拿刚采的两条跑转换，确认真实数据和 L2 的假数据结论一致。`episode_0001` 在采集时已标失败，不用手工指定：

```bash
cd ~/g1d_infra
./data_convert/convert.sh \
  --raw-dir /tmp/g1d_test_data/smoke_test \
  --output-dir /tmp/g1d_test_out
```

**成功标志**：`Auto-detected failed episodes ...: [1]`、`Converted 2 episodes, tagged bad=1`，且 `meta/episode_quality.json` 里 `(1, True)`。这一步是在真实数据上验证 L2.2b 的自动识别——采数时按的左 X 一路传到了数据集标签。

### L4.5 重放刚采的轨迹

把 L4.3 采的第一条在机器人上开环放一遍。这是「采到的数据能不能真的驱动机器人」的闭环验证。

**机械臂会自己动。清空工作区，手放急停上。**先半速：

```bash
cd ~/g1d_infra
./scripts/run_replay.sh /tmp/g1d_test_data/smoke_test/episode_0000 --dry-run
./scripts/run_replay.sh /tmp/g1d_test_data/smoke_test/episode_0000 --speed 0.5
```

| 阶段 | 预期 |
|---|---|
| dry-run | `checks  : ok`，`frames` 与 L4.3 一致 |
| 确认提示 | 打印 `ready` 和 `approach` 两段的最大关节差，差值过大就先手动把机器人摆近些 |
| 准备段 | 约 3 秒平滑移到双手抬起准备姿势，稳定后才继续 |
| 接近段 | 再用约 3 秒从准备姿势平滑移到第一帧，**不跳变** |
| 重放段 | 双臂复现录制动作，夹爪开合时机与录制一致 |
| 中途按 `Q` / Ctrl+C | 停住后 hold 约 1 秒，再回 home 放下双臂 |
| 结束 | hold 1 秒后回 home 放下 |

**成功标志**：全程无跳变、无急停触发，动作轨迹肉眼看与录制时一致。确认无误后再用 `--speed 1.0` 全速跑一遍。

```bash
rm -rf /tmp/g1d_test_data /tmp/g1d_test_out
```

---

## L5 实机 policy 回退与接管

需要 L4 全过，外加 GPU 机上真的推理服务在跑。**这一级机械臂会自己动，留足空间，手放急停上。**

### L5.1 启动

GPU 机先起推理服务，然后：

```bash
cd ~/g1d_infra
SSH_HOST=<gpu-ip> SSH_PORT=22 SSH_USER=<user> SSH_KEY=~/.ssh/id_rsa \
REMOTE_POLICY_HOST=127.0.0.1 REMOTE_POLICY_PORT=5555 \
INSTRUCTION="pick up the red cup" \
./scripts/run_deploy.sh
```

**成功标志**

```text
[launcher] Policy server is listening on 127.0.0.1:5555
[launcher] Opening SSH tunnel 127.0.0.1:15555 -> 127.0.0.1:5555
```

随后 `policy_deploy.py` 起来，Vuer 网页可进 XR。

远端没在听会明确报错并退出，这是对的：

```text
[launcher] Policy server is NOT listening on 127.0.0.1:5555.
[launcher] Local 127.0.0.1:15555 being open only means the SSH tunnel exists.
```

### L5.2 状态机全流程

**按键来自 ssh 终端 stdin，不是手柄。**

| # | 操作 | 预期现象 | 这步在验什么 |
|---|---|---|---|
| 1 | 终端按 `S` | 状态进 `POLICY_LIVE`，机械臂按 policy 动 | `POLICY_IDLE --S--> POLICY_LIVE` |
| 2 | 按 `B` | policy 停，机械臂**倒放**最近约 3 秒，然后**停在回退终点不动** | 回退缓冲；停住说明 hold 的是最后一拍命令而不是实测角 |
| 3 | 看头显 | 出现 TARGET 坐标轴 | 回退终点经 FK 转成了 OpenXR TARGET |
| 4 | 把手柄 RGB 轴对上 TARGET | 位置 ≤ 4 cm、旋转 ≤ 0.20 rad、稳 0.5 s 后提示 aligned | 对齐判据 |
| 5 | 按 `A` | 进 `TELEOP_LIVE`，机械臂跟手；前若干秒增益从 0 渐升，**不会猛冲** | 相对接管 + blend |
| 6 | 松开 A，等 `Teleoperation handoff active` | 打印该日志 | 防抖窗口结束 |
| 7 | 再按一次 `A`（或 `S`） | 从当前遥操作姿态交回 policy，**不需要重新对齐** | 交回路径 |
| 8 | 重复 2–7 一次 | 行为一致 | 状态机可复用，无残留 |
| 9 | 在 `TELEOP_LIVE` 里按 `B` | 照样进回退 | blend 期间也能中止 |
| 10 | 按 `Q` | 干净退出，隧道关闭 | `cleanup` trap |

**整体成功标志**：10 步全过，且全程没有出现——

- 回退播完后机械臂往回抽（第 2 步是核心回归点，对应 L1 的 `test_rollback_hold_uses_last_command_not_measured_state`）。
- 第一次按 `A` 后机械臂突然加速（blend 没生效）。
- 按住 `A` 被识别成「接管 + 立刻交回」（防抖失效）。
- 退出后 `15555` 端口还占着：`ss -ltn | grep 15555` 应该没输出。

### L5.3 tracking 丢失

对齐阶段把手柄放下 / 遮住，让 tracking 失效。

**成功标志**：机械臂**保持在回退终点**，第一次按 `A` 不启动接管。tracking 恢复后重新对齐即可继续。

### L5.4 边界项（可选）

| 场景 | 预期 |
|---|---|
| 在 `POLICY_IDLE` 直接按 `A` | 无反应 |
| 中途拔掉 GPU 机网线 | 报推理超时/失败并退出或停住，机械臂不乱动 |
| 加 `--record` 跑一轮 | 部署过程也写出 `episode_XXXX/`，格式同 L4.3 |

---

## 验收记录表

| 级别 | 项目 | 结果 | 备注 |
|---|---|---|---|
| L0 | `tests/check_env.py` 全 PASS | ☐ | |
| L1 | 89 个单元测试 `OK` | ☐ | |
| L2.1 | 3 条假 episode，1 条带 `FAILED` | ☐ | |
| L2.2 | `Converted 3 episodes, tagged bad=1`，v3.0 目录完整 | ☐ | |
| L2.2b | 坏轨迹三种模式结果为 `[1]` / `[1,2]` / `[]` | ☐ | |
| L2.3 | 上传 `--dry-run` 列出 10 个文件 | ☐ | |
| L2.4 | 重放 dry-run 通过，跳变数据被拒（退出码 2） | ☐ | |
| L3 | `roundtrip OK`，帧 384×320、chunk (64,16) | ☐ | |
| L4.1 | 相机服务三路可达 | ☐ | |
| L4.2 | `--no-record` 遥操作跟手，A/B 键正常 | ☐ | |
| L4.3 | 两条 episode，成功/失败标记正确 | ☐ | |
| L4.4 | 实采数据转换通过 | ☐ | |
| L4.5 | 实采轨迹重放，接近段与重放段均无跳变 | ☐ | |
| L5.1 | 隧道建立、policy 服务探测通过 | ☐ | |
| L5.2 | 状态机 10 步全过 | ☐ | |
| L5.3 | tracking 丢失时保持不动 | ☐ | |

测试机信息：机器型号 / 系统 / conda 环境名 / commit hash（`git rev-parse --short HEAD`）。

---

## 相关文档

- [README.md](README.md) 仓库总览
- [GUIDE_COLLECT.md](GUIDE_COLLECT.md) 数采操作细节与按键表
- [GUIDE_DEPLOY.md](GUIDE_DEPLOY.md) 状态机、按键、对齐参数
- [GUIDE_REPLAY.md](GUIDE_REPLAY.md) 轨迹重放参数与安全校验

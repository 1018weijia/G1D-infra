# 单人数采

`collect.py` 只做 XR 遥操作录制。没有 policy、没有回退、没有接管。

## 启动

```bash
cd ~/g1d_infra
./scripts/run_collect.sh
```

默认从 [`configs/robot_g1d.yaml`](configs/robot_g1d.yaml) 读取 `image_host` / `dds_interface` / task 默认值；环境变量仍可覆盖。

等价于：

```bash
cd ~/g1d_infra
UNITREE_DDSINTERFACE=eth0 python collect.py --ee dex1_internal --input-mode controller --task-dir ~/unitree_eai_environment/data/ --task-name test_only --task-goal "test_only"
```

录制默认开启。想只遥操作、不写盘（练手或调试）加 `--no-record`。

Rerun 可视化默认关闭，要用加 `--rerun`，且机器上得有 `DISPLAY`。它会在每次开始录制时阻塞主循环约 3.7 秒（机械臂会跳变），每帧再多花约 12 毫秒；没有显示器时自动忽略。

环境变量：`INPUT_MODE`、`TASK_DIR`、`TASK_NAME`、`TASK_GOAL`、`IMAGE_HOST`、`UNITREE_DDSINTERFACE`、`ROBOT_CONFIG`。

启动前不要同时跑其他机械臂控制程序（与 `policy_deploy` 共享 `/tmp/g1d_arm_owner.lock`）。图像服务需已在 `192.168.123.164`（可用 `IMAGE_HOST` 覆盖）。

启动后双臂先在 3 秒内平滑收到关节零位并停住，夹爪再张开、闭合、再张开。按 `R` 开始遥操作之前手臂停在零位、夹爪保持张开；开始后夹爪跟扳机。抬手或夹爪预检时按 `Q` 中止。

## 手柄按键

按键来自 Pico / OpenXR 手柄，不是 ssh 终端。

| 手柄 | 作用 |
|---|---|
| 右 A | 启动遥操作；之后在未录制时暂停 / 恢复（方便摆场景） |
| 左 Y | 开始或保存当前 episode |
| 左 X | 停止当前 episode 并标记失败 |
| 右 B | 退出 |

录制中不能暂停。保存 episode 后会自动暂停，摆完场景再按右 A 恢复。

键盘备用（sshkeyboard，未开 `--ipc` 时）：`R` / `S` / `F` / `Q` 对应上面四键。两种模式都有效，不跟手柄 A/B 抢。

## 腰部模式

默认不控腰。弯腰关节保持锁死，轻轻推不会把腰压塌，右摇杆左右也不会转腰。需要转腰或弯腰时加 `--use-waist`：

```bash
./scripts/run_collect.sh --use-waist
```

只遥操作、不写盘（练手或调试）：

```bash
./scripts/run_collect.sh --use-waist --no-record
```

`scripts/run_collect.sh` 不会自己打开这个开关。加上之后默认仍是局部模式，按键和上面一样。右手摇杆要垂直按到底、听到咔哒一声再松开，才从局部切到全身；再按一次切回来。把摇杆往下扳不是这个键，那个方向仍是升降。右摇杆左右才是转腰，而且只在全身模式生效。

| | 局部模式 | 全身模式 |
|---|---|---|
| 右 A | 启动 / 暂停 | 按住直腰，松开即停 |
| 右 B | 退出 | 按住弯腰，松开即停 |
| 右摇杆左右 | 不转腰 | 转腰 |
| 右摇杆按下 | 切到全身 | 切回局部 |
| 左 Y / 左 X | 开始或保存 / 标记失败 | 不变 |

局部模式（含刚启动、暂停、从全身切回来）弯腰锁死在当前角度。只有全身模式才放开：按住右 B 弯腰，按住右 A 直腰，松开即停。从全身切回局部只停住当前角度，不会自动直腰。退出程序时弯腰先回零，回零后再锁上。`waist.qpos` 仍然只有转腰；弯腰另记在 `waist_pitch.qpos`，转换脚本不读这个字段，不会进 policy 动作。

## Episode 目录

```text
<task-dir>/<task-name>/episode_XXXX/
  data.json
  colors/
  FAILED          # 仅失败轨迹存在
```

`data.json` 顶层有 `"success": true/false`。失败轨迹另写 `FAILED` 文件。转换时这两个标记都会被自动识别，不用手工再列一遍；`--bad` 只用来补充事后才判定为坏的轨迹。

## 之后

先做本地摘要（可选）：

```bash
python scripts/episode_summary.py ~/unitree_eai_environment/data/pick_place
```

再转换上传：

```bash
cd ~/g1d_infra/data_convert
./convert.sh --raw-dir ~/unitree_eai_environment/data/pick_place --output-dir ~/g1d_infra/datasets/pick_place --bad 3,7 # 可选：采集时没标、事后才看出来的坏轨迹
./upload.sh --token "$MODELSCOPE_API_TOKEN" --repo owner/pick_place --local-dir ~/g1d_infra/datasets/pick_place
```

`--raw-dir` 与 `--output-dir` 为必填。

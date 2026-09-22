# 单人数采

`collect.py` 只做 XR 遥操作录制。没有 policy、没有回退、没有接管。

## 启动

```bash
cd ~/g1d_infra
./scripts/run_collect.sh
```

等价于：

```bash
cd ~/g1d_infra
UNITREE_DDSINTERFACE=eth0 python collect.py \
  --ee dex1_internal \
  --input-mode controller \
  --task-dir ~/unitree_eai_environment/data/ \
  --task-name pick_place \
  --task-goal "pick and place"
```

录制默认开启。想只遥操作、不写盘（练手或调试）加 `--no-record`。

Rerun 可视化默认关闭，要用加 `--rerun`，且机器上得有 `DISPLAY`。它会在每次开始录制时阻塞主循环约 3.7 秒（机械臂会跳变），每帧再多花约 12 毫秒；没有显示器时自动忽略。

环境变量：`EE`、`INPUT_MODE`、`TASK_DIR`、`TASK_NAME`、`TASK_GOAL`、`IMAGE_HOST`、`UNITREE_DDSINTERFACE`。

启动前不要同时跑其他机械臂控制程序。图像服务需已在 `192.168.123.164`（可用 `IMAGE_HOST` 覆盖）。

## 手柄按键

按键来自 Pico / OpenXR 手柄，不是 ssh 终端。

| 手柄 | 作用 |
|---|---|
| 右 A | 启动遥操作；之后在未录制时暂停 / 恢复（方便摆场景） |
| 左 Y | 开始或保存当前 episode |
| 左 X | 停止当前 episode 并标记失败 |
| 右 B | 退出 |

录制中不能暂停。保存 episode 后会自动暂停，摆完场景再按右 A 恢复。

键盘备用（sshkeyboard，未开 `--ipc` 时）：`R` / `S` / `F` / `Q` 对应上面四键。

## Episode 目录

```text
<task-dir>/<task-name>/episode_XXXX/
  data.json
  colors/
  FAILED          # 仅失败轨迹存在
```

`data.json` 顶层有 `"success": true/false`。失败轨迹另写 `FAILED` 文件。转换时这两个标记都会被自动识别，不用手工再列一遍；`--bad` 只用来补充事后才判定为坏的轨迹。

## 之后

```bash
cd ~/g1d_infra/data_convert
./convert.sh --raw-dir ~/unitree_eai_environment/data/pick_place \
             --output-dir ~/g1d_infra/datasets/pick_place \
             --bad 3,7      # 可选：采集时没标、事后才看出来的坏轨迹
./upload.sh --token "$MODELSCOPE_API_TOKEN" --repo owner/pick_place \
            --local-dir ~/g1d_infra/datasets/pick_place
```

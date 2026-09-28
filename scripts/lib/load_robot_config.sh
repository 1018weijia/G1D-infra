#!/usr/bin/env bash
# Load configs/robot_g1d.yaml into G1D_* shell variables (env still wins later).

g1d_load_robot_config() {
  local repo_root script_dir config_path
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "${script_dir}/../.." && pwd)"
  config_path="${ROBOT_CONFIG:-${repo_root}/configs/robot_g1d.yaml}"

  if [[ ! -r "${config_path}" ]]; then
    echo "[config] robot config not readable: ${config_path}" >&2
    return 1
  fi

  eval "$(
    python3 - "${config_path}" "${repo_root}" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

path = Path(sys.argv[1])
repo = Path(sys.argv[2])
data = {}
if yaml is not None:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
else:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")

def resolve(value):
    text = str(value)
    if not text.startswith("/"):
        return str(repo / text)
    return text

mapping = {
    "IMAGE_HOST": data.get("image_host", "192.168.123.164"),
    "DDS_INTERFACE": data.get("dds_interface", "eth0"),
    "TASK_DIR": data.get("task_dir", str(Path.home() / "unitree_eai_environment/data/")),
    "TASK_NAME": data.get("task_name", "pick_place"),
    "TASK_GOAL": data.get("task_goal", "pick and place"),
    "READY_POSE_CONFIG": resolve(data.get("ready_pose_config", "configs/ready_pose.json")),
    "ALIGNMENT_TARGETS": resolve(data.get("alignment_targets", "configs/alignment_targets.json")),
    "LOCAL_POLICY_HOST": data.get("local_policy_host", "127.0.0.1"),
    "LOCAL_POLICY_PORT": data.get("local_policy_port", 15555),
    "POLICY_PROTOCOL": data.get("policy_protocol", "zmq"),
}
for key, value in mapping.items():
    print(f"export G1D_{key}={str(value)!r}")
PY
  )"
}

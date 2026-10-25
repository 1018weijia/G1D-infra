#!/usr/bin/env bash
# 检测当前是否有程序正在控制机械臂；有的话询问是否结束它。
#
#   ./scripts/stop_control.sh              # 检测，有控制者就问 y/n
#   ./scripts/stop_control.sh -y           # 不问，直接结束
#   ./scripts/stop_control.sh -n           # 只检测，不管它
#   ./scripts/stop_control.sh --probe-only # 只报告，不打印询问
#
# 退出码：0 无控制者 / 2 仍有控制者 / 3 用户选择不管 / 1 出错
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PY_SCRIPT="$SCRIPT_DIR/stop_arm_control.py"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

# Pick an interpreter that can import unitree_sdk2py; fall back with a clear error.
pick_python() {
  local candidate
  for candidate in "${G1D_PYTHON:-}" python python3; do
    [[ -n "$candidate" ]] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c 'import unitree_sdk2py' >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON="$(pick_python || true)"
if [[ -z "$PYTHON" ]]; then
  echo "[stop_control] 找不到能 import unitree_sdk2py 的 python。" >&2
  echo "[stop_control] 请先激活遥操作环境（conda activate tv），或用 G1D_PYTHON 指定解释器。" >&2
  exit 1
fi

cd "$REPO_ROOT"
exec "$PYTHON" "$PY_SCRIPT" "$@"

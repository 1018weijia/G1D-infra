#!/usr/bin/env bash
# Upload a local LeRobot dataset folder to a ModelScope dataset repo.
#
# Usage:
#   ./upload.sh --token ms-xxxx --repo username/pick_place_100 \
#               --local-dir /home/unitree/g1d_infra/datasets/pick_place_100
#   ./upload.sh ms-xxxx username/pick_place_100 \
#               /home/unitree/data2lerobot/datasets/pick_place_100
#   MODELSCOPE_API_TOKEN=ms-xxxx ./upload.sh --repo username/pick_place_100 \
#               --local-dir ~/data2lerobot/datasets/pick_place_100
#
# Token: ModelScope access token from https://www.modelscope.cn/my/myaccesstoken
# Repo : username/dataset_name  (or just dataset_name; username is inferred)
# Path : converted LeRobot folder, e.g. datasets/pick_place_100

set -euo pipefail

UNITREE_PYTHON="${HOME}/miniconda3/envs/unitree_lerobot/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "${UNITREE_PYTHON}" ]]; then
  PYTHON_BIN="${UNITREE_PYTHON}"
else
  PYTHON_BIN="python3"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPLOAD_PY="${SCRIPT_DIR}/upload_to_modelscope.py"

usage() {
  cat <<'EOF'
Upload a local LeRobot dataset to a ModelScope dataset repository.

Usage:
  ./upload.sh --token TOKEN --repo OWNER/NAME --local-dir PATH
  ./upload.sh TOKEN OWNER/NAME PATH

Options:
  --token TOKEN           ModelScope access token (or MODELSCOPE_API_TOKEN)
  --repo OWNER/NAME       Dataset repo id, e.g. alice/pick_place_100
  --local-dir PATH        Local dataset folder
  --private               Create/use a private repo (default: public)
  --include-images        Also upload images/ (skipped by default)
  --max-workers N         Parallel upload workers (default: 4)
  --commit-message TEXT   Commit message
  --dry-run               List files, do not upload
  --no-create             Do not create the remote repo if missing
  --endpoint URL          ModelScope endpoint
  -h, --help              Show this help

Examples:
  ./upload.sh --token ms-xxxx --repo alice/pick_place_100 \
      --local-dir /home/unitree/data2lerobot/datasets/pick_place_100

  ./upload.sh ms-xxxx alice/pick_place_100 \
      /home/unitree/data2lerobot/datasets/pick_place_100
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
  esac
done

if [[ ! -f "${UPLOAD_PY}" ]]; then
  echo "Missing uploader: ${UPLOAD_PY}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import modelscope_hub" >/dev/null 2>&1 \
  && ! "${PYTHON_BIN}" -c "import modelscope" >/dev/null 2>&1; then
  echo "Python at ${PYTHON_BIN} is missing ModelScope SDK." >&2
  echo "Install with:" >&2
  echo "  ${PYTHON_BIN} -m pip install modelscope-hub" >&2
  echo "or:" >&2
  echo "  ${PYTHON_BIN} -m pip install modelscope" >&2
  exit 1
fi

echo "Python: ${PYTHON_BIN}"
exec "${PYTHON_BIN}" "${UPLOAD_PY}" "$@"

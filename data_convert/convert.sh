#!/usr/bin/env bash
# Convert a Unitree JSON dataset (episode_XXXX/data.json) to LeRobot v3.0.
#
# Usage:
#   ./convert.sh
#   ./convert.sh --raw-dir ~/unitree_eai_environment/data/pick_place_100 \
#                --output-dir ~/g1d_infra/datasets/pick_place_100
#   ./convert.sh --bad 7,15,23
#   ./convert.sh --skip-bad          # drop bad episodes instead of tagging them
#
# Fill BAD_EPISODES below, then run the script. Numbers match episode_XXXX
# folders: episode_0007 -> 7.

set -euo pipefail

# =============================================================================
# 坏轨迹列表（必填位置）
# 填写采集目录里的轨迹序号，对应 episode_XXXX 文件夹的数字。
# 例：episode_0007、episode_0015 写成 7 15 或 7,15
# 留空表示全部标为好数据。
# =============================================================================
BAD_EPISODES=""

INPUT="${HOME}/unitree_eai_environment/data/pick_place_100"
OUTPUT="${HOME}/g1d_infra/datasets/pick_place_100"
ROBOT_TYPE="Unitree_G1_MoveibleLift_Dex1_NoUseWaist"
TASK=""
SKIP_BAD=0
OVERWRITE=1
IMAGE_WRITER_THREADS=""
IMAGE_WRITER_PROCESSES=""
BATCH_ENCODING_SIZE=""
MAX_EPISODES=""
VCODEC=""
PRESET=""
STREAMING_ENCODING=""
ENCODER_THREADS=""
ENCODER_QUEUE_MAXSIZE=""
IMAGE_LOAD_WORKERS=""
SCHEMA_EPISODES=""
UNITREE_PYTHON="${HOME}/miniconda3/envs/unitree_lerobot/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "${UNITREE_PYTHON}" ]]; then
  PYTHON_BIN="${UNITREE_PYTHON}"
else
  PYTHON_BIN="python3"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONSTANTS="${CONSTANTS:-${SCRIPT_DIR}/constants.py}"
LEROBOT_SRC="${LEROBOT_SRC:-${REPO_ROOT}/3rd/lerobot/src}"

usage() {
  cat <<'EOF'
Convert Unitree JSON episodes to a local LeRobot v3.0 dataset.

Options:
  --raw-dir PATH                JSON task dir (contains episode_XXXX)
  --output-dir PATH             LeRobot dataset output dir
  --constants PATH              robot constants.py (default: this directory)
  --robot-type NAME             robot_type key in constants.py
  --bad IDS                     bad episode numbers, e.g. 7,15 or "7 15"
  --task TEXT                   override language instruction
  --skip-bad                    do not convert bad episodes
  --no-overwrite                fail if output already exists
  --max-episodes N              convert only the first N episodes
  --vcodec NAME                 h264 (fast, default), libsvtav1 (smaller), auto
  --preset NAME                 libx264 preset, default veryfast
  --no-streaming-encoding       PNG round-trip instead of live MP4 encode
  --encoder-threads N           threads per camera encoder (default 1)
  --image-load-workers N        parallel JPEG readers (default 4)
  --image-writer-threads N      async PNG writer threads
  --image-writer-processes N    async PNG writer processes
  --batch-encoding-size N       ignored; encoding is per episode
  -h, --help                    show this help

--input/--output are aliases for --raw-dir/--output-dir.
Edit BAD_EPISODES at the top of this script to keep a persistent bad list.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input|--raw-dir)
      INPUT="$2"
      shift 2
      ;;
    --output|--output-dir)
      OUTPUT="$2"
      shift 2
      ;;
    --robot-type)
      ROBOT_TYPE="$2"
      shift 2
      ;;
    --bad|--bad-episodes)
      BAD_EPISODES="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --skip-bad)
      SKIP_BAD=1
      shift
      ;;
    --no-overwrite)
      OVERWRITE=0
      shift
      ;;
    --max-episodes)
      MAX_EPISODES="$2"
      shift 2
      ;;
    --image-writer-threads)
      IMAGE_WRITER_THREADS="$2"
      shift 2
      ;;
    --image-writer-processes)
      IMAGE_WRITER_PROCESSES="$2"
      shift 2
      ;;
    --batch-encoding-size|--batch_encoding_size)
      BATCH_ENCODING_SIZE="$2"
      shift 2
      ;;
    --vcodec)
      VCODEC="$2"
      shift 2
      ;;
    --preset)
      PRESET="$2"
      shift 2
      ;;
    --streaming-encoding)
      STREAMING_ENCODING=1
      shift
      ;;
    --no-streaming-encoding)
      STREAMING_ENCODING=0
      shift
      ;;
    --encoder-threads)
      ENCODER_THREADS="$2"
      shift 2
      ;;
    --encoder-queue-maxsize)
      ENCODER_QUEUE_MAXSIZE="$2"
      shift 2
      ;;
    --image-load-workers)
      IMAGE_LOAD_WORKERS="$2"
      shift 2
      ;;
    --schema-episodes)
      SCHEMA_EPISODES="$2"
      shift 2
      ;;
    --constants)
      CONSTANTS="$2"
      shift 2
      ;;
    --unitree-root)
      echo "--unitree-root was removed; robot constants now come from this repo." >&2
      echo "Default: ${SCRIPT_DIR}/constants.py" >&2
      echo "To point elsewhere, pass a constants.py path: --constants PATH" >&2
      exit 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

INPUT="${INPUT/#\~/${HOME}}"
OUTPUT="${OUTPUT/#\~/${HOME}}"

if [[ ! -d "${INPUT}" ]]; then
  echo "Input directory not found: ${INPUT}" >&2
  exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/convert_unitree_json.py" ]]; then
  echo "Missing converter: ${SCRIPT_DIR}/convert_unitree_json.py" >&2
  exit 1
fi

if [[ ! -d "${LEROBOT_SRC}/lerobot" ]]; then
  echo "Official lerobot source not found at ${LEROBOT_SRC}" >&2
  echo "Initialize the submodule: git submodule update --init --recursive" >&2
  exit 1
fi

export PYTHONPATH="${LEROBOT_SRC}:${PYTHONPATH:-}"

if ! "${PYTHON_BIN}" -c "import huggingface_hub, datasets, av, torch" >/dev/null 2>&1; then
  echo "Python at ${PYTHON_BIN} is missing LeRobot dataset deps (huggingface_hub, datasets, av, torch)." >&2
  echo "Activate the unitree_lerobot env, or run:" >&2
  echo "  PYTHON_BIN=${UNITREE_PYTHON} ./convert.sh ..." >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/convert_unitree_json.py"
  --raw-dir "${INPUT}"
  --output-dir "${OUTPUT}"
  --robot-type "${ROBOT_TYPE}"
  --constants "${CONSTANTS}"
)

if [[ -n "${BAD_EPISODES}" ]]; then
  CMD+=(--bad-episodes "${BAD_EPISODES}")
fi

if [[ -n "${TASK}" ]]; then
  CMD+=(--task "${TASK}")
fi

if [[ "${SKIP_BAD}" -eq 1 ]]; then
  CMD+=(--skip-bad)
fi

if [[ "${OVERWRITE}" -eq 1 ]]; then
  CMD+=(--overwrite)
fi

if [[ -n "${MAX_EPISODES}" ]]; then
  CMD+=(--max-episodes "${MAX_EPISODES}")
fi

if [[ -n "${IMAGE_WRITER_THREADS}" ]]; then
  CMD+=(--image-writer-threads "${IMAGE_WRITER_THREADS}")
fi

if [[ -n "${IMAGE_WRITER_PROCESSES}" ]]; then
  CMD+=(--image-writer-processes "${IMAGE_WRITER_PROCESSES}")
fi

if [[ -n "${BATCH_ENCODING_SIZE}" ]]; then
  CMD+=(--batch-encoding-size "${BATCH_ENCODING_SIZE}")
fi

if [[ -n "${VCODEC}" ]]; then
  CMD+=(--vcodec "${VCODEC}")
fi

if [[ -n "${PRESET}" ]]; then
  CMD+=(--preset "${PRESET}")
fi

if [[ "${STREAMING_ENCODING}" == "1" ]]; then
  CMD+=(--streaming-encoding)
elif [[ "${STREAMING_ENCODING}" == "0" ]]; then
  CMD+=(--no-streaming-encoding)
fi

if [[ -n "${ENCODER_THREADS}" ]]; then
  CMD+=(--encoder-threads "${ENCODER_THREADS}")
fi

if [[ -n "${ENCODER_QUEUE_MAXSIZE}" ]]; then
  CMD+=(--encoder-queue-maxsize "${ENCODER_QUEUE_MAXSIZE}")
fi

if [[ -n "${IMAGE_LOAD_WORKERS}" ]]; then
  CMD+=(--image-load-workers "${IMAGE_LOAD_WORKERS}")
fi

if [[ -n "${SCHEMA_EPISODES}" ]]; then
  CMD+=(--schema-episodes "${SCHEMA_EPISODES}")
fi

echo "Input : ${INPUT}"
echo "Output: ${OUTPUT}"
echo "Robot : ${ROBOT_TYPE}"
echo "Python: ${PYTHON_BIN}"
echo "Bad   : ${BAD_EPISODES:-<none>}"
echo

exec "${CMD[@]}"

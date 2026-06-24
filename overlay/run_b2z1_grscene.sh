#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

SCENE="${SCENE:-mini_home}"
STEPS="${STEPS:-0}"
FORWARD_SPEED="${FORWARD_SPEED:-0.03}"
LATERAL_SPEED="${LATERAL_SPEED:-0.0}"
ROTATION_SPEED="${ROTATION_SPEED:-0.0}"
SCENE_POSITION="${SCENE_POSITION:-0,0,0}"
SCENE_RENDER_WARMUP_SECONDS="${SCENE_RENDER_WARMUP_SECONDS:-0}"
STABILIZE_SECONDS="${STABILIZE_SECONDS:-5.0}"
RAMP_SECONDS="${RAMP_SECONDS:-8.0}"
MOVE_SECONDS="${MOVE_SECONDS:-0}"
HEADING_HOLD="${HEADING_HOLD:-1}"
MAX_HEADING_CORRECTION="${MAX_HEADING_CORRECTION:-0.08}"
STABILIZE_MODE="${STABILIZE_MODE:-hold}"

if [[ -z "${SCENE_SCALE:-}" ]]; then
  case "${SCENE}" in
    empty)
      SCENE_SCALE="1,1,1"
      ;;
    mini_home)
      SCENE_SCALE="1,1,1"
      ;;
    *)
      # GRScenes assets use centimeter-scale coordinates; Isaac/B2Z1 uses meters.
      SCENE_SCALE="0.01,0.01,0.01"
      ;;
  esac
fi

if [[ -z "${ROBOT_POSITION:-}" ]]; then
  case "${SCENE}" in
    home)
      # Center of the largest ground patch in MWBGLKQKTKJZ2AABAAAAACA8_usd after 0.01 scaling.
      ROBOT_POSITION="-1.42,-2.71,0.6"
      ;;
    mini_home)
      ROBOT_POSITION="0,0,0.6"
      ;;
    *)
      ROBOT_POSITION="0,0,0.6"
      ;;
  esac
fi

DEMO_ARGS=(
  --scene "${SCENE}"
  --steps "${STEPS}"
  --forward-speed "${FORWARD_SPEED}"
  --lateral-speed "${LATERAL_SPEED}"
  --rotation-speed "${ROTATION_SPEED}"
  --stabilize-seconds "${STABILIZE_SECONDS}"
  --stabilize-mode "${STABILIZE_MODE}"
  --ramp-seconds "${RAMP_SECONDS}"
  --max-heading-correction "${MAX_HEADING_CORRECTION}"
  "--robot-position=${ROBOT_POSITION}"
  "--scene-position=${SCENE_POSITION}"
  "--scene-scale=${SCENE_SCALE}"
)

if [[ "${HEADING_HOLD}" == "0" ]]; then
  DEMO_ARGS+=(--no-heading-hold)
fi

if [[ "${STATIC_SCENE_BACKGROUND:-0}" == "1" ]]; then
  DEMO_ARGS+=(--static-scene-background)
fi

if [[ "${IDLE:-0}" == "1" ]]; then
  DEMO_ARGS+=(--idle)
fi

if [[ "${MOVE_SECONDS}" != "0" ]]; then
  DEMO_ARGS+=(--move-seconds "${MOVE_SECONDS}")
fi

if [[ "${SCENE_RENDER_WARMUP_SECONDS}" != "0" ]]; then
  DEMO_ARGS+=(--scene-render-warmup-seconds "${SCENE_RENDER_WARMUP_SECONDS}")
fi

if [[ "${HEADLESS:-0}" == "1" ]]; then
  DEMO_ARGS+=(--headless)
fi

exec ./run_internutopia_docker.sh -lc '
set -euo pipefail
source /isaac-sim/.venv/bin/activate
set +u
source /isaac-sim/python.env.init
set -u
cd /isaac-sim/InternUtopia
python internutopia/demo/b2z1_locomotion.py "$@"
' bash "${DEMO_ARGS[@]}" "$@"

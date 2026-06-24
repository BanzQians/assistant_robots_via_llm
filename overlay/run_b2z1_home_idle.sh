#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

IDLE="${IDLE:-1}" \
SCENE_RENDER_WARMUP_SECONDS="${SCENE_RENDER_WARMUP_SECONDS:-20}" \
SCENE="${SCENE:-home}" \
ROBOT_POSITION="${ROBOT_POSITION:--0.54,0.22,0.6}" \
FORWARD_SPEED="${FORWARD_SPEED:-0.0}" \
LATERAL_SPEED="${LATERAL_SPEED:-0.0}" \
ROTATION_SPEED="${ROTATION_SPEED:-0.0}" \
STEPS="${STEPS:-0}" \
./run_b2z1_grscene.sh \
  --scene-asset-path "${SCENE_ASSET_PATH:-/isaac-sim/InternUtopia/internutopia/assets_full/scenes/GRScenes-100/home_scenes/scenes/MWAX5JYKTKJZ2AABAAAAAEA8_usd/start_result_navigation.usd}"

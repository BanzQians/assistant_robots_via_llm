#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

DEFAULT_IMAGE="internutopia-b2z1-pin:2.2.0"
FALLBACK_IMAGE="registry.cn-hangzhou.aliyuncs.com/internutopia/internutopia:2.2.0"
IMAGE="${INTERNUTOPIA_IMAGE:-${DEFAULT_IMAGE}}"
CACHE_ROOT="${HOME}/isaac-sim-cache"
DOCKER_NAME="${INTERNUTOPIA_CONTAINER_NAME:-internutopia}"
GRSCENES_MDL_SYSTEM_PATH="/isaac-sim/InternUtopia/internutopia/assets_full/scenes/GRScenes-100/home_scenes/Materials"
GRSCENES_MDL_SYSTEM_PATH+=":/isaac-sim/InternUtopia/internutopia/assets_full/scenes/GRScenes-100/commercial_scenes/Materials"
MDL_SYSTEM_PATH_CONTAINER="${MDL_SYSTEM_PATH:+${MDL_SYSTEM_PATH}:}${GRSCENES_MDL_SYSTEM_PATH}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not in PATH."
  exit 1
fi

if [ -z "${INTERNUTOPIA_IMAGE:-}" ] && ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  IMAGE="${FALLBACK_IMAGE}"
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi failed on the host. Fix the NVIDIA driver before running Isaac Sim."
  exit 1
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "NVIDIA Container Toolkit is missing. Installing it now..."
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends curl ca-certificates gnupg

  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
fi

if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  echo "Configuring Docker NVIDIA runtime..."
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

mkdir -p \
  "${CACHE_ROOT}/kit" \
  "${CACHE_ROOT}/ov" \
  "${CACHE_ROOT}/pip" \
  "${CACHE_ROOT}/glcache" \
  "${CACHE_ROOT}/computecache" \
  "${CACHE_ROOT}/logs" \
  "${CACHE_ROOT}/data" \
  "${CACHE_ROOT}/documents"

xhost +local:root

DOCKER_RUN_MODE=()
if [ "${INTERNUTOPIA_DETACH:-0}" = "1" ]; then
  DOCKER_RUN_MODE=(-d)
elif [ -t 0 ]; then
  DOCKER_RUN_MODE=(-it)
fi

exec docker run --name "${DOCKER_NAME}" "${DOCKER_RUN_MODE[@]}" --rm --gpus all --network host \
  --ipc=host \
  --shm-size=16g \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -e "DISPLAY=${DISPLAY:-:0}" \
  -e "NVIDIA_DRIVER_CAPABILITIES=all" \
  -e "INTERNUTOPIA_ASSETS_PATH=/isaac-sim/InternUtopia/internutopia/assets_full" \
  -e "MDL_SYSTEM_PATH=${MDL_SYSTEM_PATH_CONTAINER}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "${PWD}:/isaac-sim/InternUtopia" \
  -v "${CACHE_ROOT}/kit:/isaac-sim/kit/cache:rw" \
  -v "${CACHE_ROOT}/ov:/root/.cache/ov:rw" \
  -v "${CACHE_ROOT}/pip:/root/.cache/pip:rw" \
  -v "${CACHE_ROOT}/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "${CACHE_ROOT}/computecache:/root/.nv/ComputeCache:rw" \
  -v "${CACHE_ROOT}/logs:/root/.nvidia-omniverse/logs:rw" \
  -v "${CACHE_ROOT}/data:/root/.local/share/ov/data:rw" \
  -v "${CACHE_ROOT}/documents:/root/Documents:rw" \
  "${IMAGE}" "$@"

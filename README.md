# Assistant Robots via LLM - InternUtopia Overlay

This repository intentionally does not vendor the full InternUtopia project.
It contains only our additions on top of InternUtopia:

- LLM/mock task simulation demos.
- A GRScenes inspection helper.
- The custom B2Z1 + Z1 arm extension, local assets, and locomotion policy.
- A small patch for upstream InternUtopia files that need registration or runtime tweaks.

## Layout

- `demos/`: root-level demo scripts to copy into an InternUtopia checkout.
- `tools/play_scene_viewer.py`: GRScenes USD viewer with a debug camera.
- `overlay/`: path-preserving files that should be copied onto InternUtopia.
- `overlay/local_assets/policies/b2z1/model_57000.pt`: B2Z1 locomotion policy.
- `overlay/local_assets/robots/b2z1/`: B2Z1 USD/mesh assets.
- `overlay/local_assets/robots/z1/`: Unitree Z1 URDF/mesh assets for IK.
- `overlay/local_assets/objects/alore/`: small local grasp demo objects.
- `patches/internutopia_b2z1_integration.patch`: edits to existing upstream files.

Large GRScenes/full-scene assets are not included:

```text
internutopia/assets_full/
```

## Git LFS

The B2Z1 and Z1 assets include files larger than GitHub's normal file limit, so
Git LFS is required before cloning or pushing this repository:

```bash
git lfs install
git lfs pull
```

The overlay assets are about 526 MB in the working tree.

## Install Into InternUtopia

From a clean or existing InternUtopia checkout:

```bash
cd /path/to/InternUtopia
git apply /path/to/assistant_robots_via_llm/patches/internutopia_b2z1_integration.patch
rsync -a /path/to/assistant_robots_via_llm/overlay/ ./
cp /path/to/assistant_robots_via_llm/demos/*.py ./
cp /path/to/assistant_robots_via_llm/tools/play_scene_viewer.py toolkits/grscenes_scripts/
```

Build the Docker image with the pinned Pinocchio dependency:

```bash
docker build -t internutopia-b2z1-pin:2.2.0 -f docker/b2z1-pin.Dockerfile .
```

Run the no-LLM mock task simulation:

```bash
docker run --rm --gpus all --network host --ipc=host --shm-size=16g \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e INTERNUTOPIA_ASSETS_PATH=/isaac-sim/InternUtopia/internutopia/assets_full \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD":/isaac-sim/InternUtopia \
  -w /isaac-sim/InternUtopia \
  internutopia-b2z1-pin:2.2.0 \
  -lc 'source /isaac-sim/.venv/bin/activate && . /isaac-sim/python.env.init && python mock_sim.py'
```

Run a small B2Z1 smoke test without GRScenes:

```bash
HEADLESS=1 STEPS=500 SCENE=mini_home ./run_b2z1_grscene.sh
```

## Notes

- `demos/task_interpreter.py` contains a machine-specific GGUF model path; change it before running the LLM-backed demo elsewhere.
- `mock_sim.py` removes the LLM dependency but still launches Isaac Sim.
- Use `SCENE=home` or `SCENE=commercial` only after separately installing/copying GRScenes into `internutopia/assets_full/`.

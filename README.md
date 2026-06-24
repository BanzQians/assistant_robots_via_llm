# Assistant Robots via LLM - InternUtopia Overlay

This repository intentionally does not vendor the full InternUtopia project.
It only contains our demo scripts, one scene viewer helper, and a small patch
for modified upstream InternUtopia files.

## Contents

- `demos/task_interpreter.py`: local llama.cpp based instruction-to-JSON planner.
- `demos/task_dispatcher.py`: JSON task plan to symbolic waypoint actions.
- `demos/llm_task_sim.py`: LLM-backed task simulation entrypoint.
- `demos/mock_sim.py`: no-LLM demo entrypoint using hardcoded task plans.
- `tools/play_scene_viewer.py`: GRScenes USD viewer with debug camera setup.
- `patches/b2z1_policy_stability.patch`: B2Z1 controller and solver/gain edits.

## Install Into InternUtopia

From an existing InternUtopia checkout:

```bash
cp /path/to/assistant_robots_via_llm/demos/*.py /path/to/InternUtopia/
cp /path/to/assistant_robots_via_llm/tools/play_scene_viewer.py \
  /path/to/InternUtopia/toolkits/grscenes_scripts/
cd /path/to/InternUtopia
git apply /path/to/assistant_robots_via_llm/patches/b2z1_policy_stability.patch
```

The scripts assume the InternUtopia Docker/Isaac Sim environment is available.
For example:

```bash
cd /path/to/InternUtopia
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

## Notes

- Large simulator assets are excluded on purpose.
- The GGUF model path in `demos/task_interpreter.py` is machine-specific and
  should be changed before running the LLM-backed demo elsewhere.
- `mock_sim.py` still launches Isaac Sim; it only removes the LLM dependency.

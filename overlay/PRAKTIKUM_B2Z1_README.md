# Praktikum B2Z1 / Arm Demos

This branch adds self-contained B2Z1 and arm-control demos on top of InternUtopia.
For running the Utopia demos, `ALORE_Legged_Manipulator` is no longer required next
to this repository.

## Included Local Assets

The required ALORE runtime resources are copied into `local_assets/`:

```text
local_assets/policies/b2z1/model_57000.pt
local_assets/robots/b2z1/urdf/b2z1_nolidar/
local_assets/robots/z1/
local_assets/objects/alore/object_1/
local_assets/objects/alore/object_4/
local_assets/objects/alore/object_7/
```

Large full-scene assets are not included in Git:

```text
internutopia/assets_full/
```

Use `SCENE=mini_home` or `SCENE=empty` for a small local smoke test. Use
`SCENE=home` or `SCENE=commercial` only after installing/copying GRScenes assets.

## Git LFS

Install Git LFS before adding or pulling the local binary assets:

```bash
git lfs install
```

This repo includes `.gitattributes` patterns for `local_assets` binary files.

## Runtime Options

Docker is recommended for reproducing this branch exactly, but it is not the
only supported path. A local Isaac Sim 4.5 + conda environment can run the same
Python demos.

### Docker

Build the image:

```bash
docker build -t internutopia-b2z1-pin:2.2.0 -f docker/b2z1-pin.Dockerfile .
```

Run B2Z1 locomotion through the wrapper:

GUI:

```bash
SCENE=mini_home FORWARD_SPEED=0.03 ./run_b2z1_grscene.sh
```

Headless smoke test:

```bash
HEADLESS=1 STEPS=500 SCENE=mini_home ./run_b2z1_grscene.sh
```

Run B2Z1 asset load:

```bash
./run_internutopia_docker.sh -lc 'source /isaac-sim/.venv/bin/activate && source /isaac-sim/python.env.init && cd /isaac-sim/InternUtopia && python internutopia/demo/b2z1_asset_load.py'
```

Run B2Z1 grasp demo:

```bash
./run_internutopia_docker.sh -lc 'source /isaac-sim/.venv/bin/activate && source /isaac-sim/python.env.init && cd /isaac-sim/InternUtopia && python internutopia/demo/b2z1_grasp_demo.py --object box'
```

Other objects:

```bash
./run_internutopia_docker.sh -lc 'source /isaac-sim/.venv/bin/activate && source /isaac-sim/python.env.init && cd /isaac-sim/InternUtopia && python internutopia/demo/b2z1_grasp_demo.py --object chair'
```

### Local Isaac Sim + Conda

Prerequisites:

- NVIDIA Omniverse Isaac Sim 4.5 installed locally.
- Conda.
- Git LFS assets pulled.

From the InternUtopia root, create the local environment:

```bash
bash setup_conda.sh
conda activate internutopia
```

`setup_conda.sh` asks for the local Isaac Sim directory containing
`isaac-sim.sh`, creates a matching Python conda env, and installs this checkout.

Run the same demos directly:

```bash
python internutopia/demo/b2z1_asset_load.py
python internutopia/demo/b2z1_locomotion.py --scene mini_home --headless --steps 500
python internutopia/demo/b2z1_grasp_demo.py --object box --headless
```

For GUI mode, remove `--headless` and use a machine with a working display.

## Upload Checklist

Do upload:

```bash
git add .gitattributes PRAKTIKUM_B2Z1_README.md
git add run_internutopia_docker.sh run_b2z1_grscene.sh docker/b2z1-pin.Dockerfile
git add internutopia/demo/b2z1_*.py internutopia/demo/b2z1_tasks_example.json
git add internutopia_extension/configs/robots/b2z1.py
git add internutopia_extension/configs/controllers/b2_move_by_speed_controller.py
git add internutopia_extension/controllers/b2_move_by_speed_controller.py
git add internutopia_extension/controllers/models/b2 internutopia_extension/controllers/z1_ik_solver.py
git add internutopia_extension/robots/b2z1.py
git add local_assets/robots/b2z1 local_assets/robots/z1 local_assets/objects/alore local_assets/policies/b2z1
```

Do not upload:

```text
internutopia/assets_full/
__pycache__/
*.pyc
.vscode/
```

"""
llm_task_sim.py

Main simulation script. Wires together:
  - task_interpreter.py  (LLM → JSON task plan)
  - task_dispatcher.py   (JSON → ordered RobotAction frames)
  - One H1 robot in GRUtopia, standing in for navigator + fetcher
    (GRUtopia's Env currently only supports a single agent per episode;
     see https://github.com/InternRobotics/InternUtopia/issues/59)

Flow:
  1. LLM interprets a mock instruction → task plan JSON
  2. Dispatcher converts plan → frames (each frame = one or more robot actions)
  3. Frames are flattened into a sequential action list, executed one after
     another by the single H1 robot. The "robot" label (navigator/fetcher)
     from each action is logged for clarity, even though only one physical
     robot exists in the sim right now.

Run:
    CUDA_VISIBLE_DEVICES=1 python llm_task_sim.py

Change MOCK_INSTRUCTION at the top to test different tasks.
"""

import json

# ── Import our modules ────────────────────────────────────────────────────────
# These must be in the same directory (InternUtopia root)
from task_interpreter import interpret, MODEL_PATH
from task_dispatcher import dispatch, print_dispatch_plan

# ── GRUtopia imports ──────────────────────────────────────────────────────────
from llama_cpp import Llama
from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia.macros import gm
from internutopia_extension import import_extensions
from internutopia_extension.configs.robots.h1 import (
    H1RobotCfg,
    move_along_path_cfg,
    move_by_speed_cfg,
    rotate_cfg,
)
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg

# ── Config ────────────────────────────────────────────────────────────────────

# Change this to test different instructions
MOCK_INSTRUCTION = "Where is the nearest bathroom?"

# Steps to wait for a robot to reach its waypoint before moving to next frame.
# 240 steps ≈ 1 second at physics_dt=1/240.
FRAME_TIMEOUT_STEPS = 720   # ~3 seconds per frame max

# How close (metres) the robot needs to be to consider waypoint reached.
ARRIVAL_THRESHOLD = 0.5

# ── Robot name constant ────────────────────────────────────────────────────────
# NOTE: GRUtopia's Env/gym_env currently only supports a single agent per
# episode (see https://github.com/InternRobotics/InternUtopia/issues/59).
# We simulate "two robots" with a single H1 that switches role/identity
# between frames depending on which robot the task plan assigns each step to.
# The interpreter + dispatcher still reason about two distinct robots; only
# the physical sim drops to one actor for now.
ROBOT_NAME = "h1"

# ── Step 1: Load LLM and interpret instruction ────────────────────────────────

print(f"\n{'='*60}")
print(f"Instruction: {MOCK_INSTRUCTION}")
print(f"{'='*60}\n")

print("[LLM] Loading model ...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=False,
)
print("(1) Model loaded.")
task_plan = interpret(llm, MOCK_INSTRUCTION)

if task_plan is None:
    print("[ERROR] LLM failed to produce a valid task plan. Exiting.")
    exit(1)

print("\n[Task Plan]")
print(json.dumps(task_plan, indent=2))

# ── Step 2: Dispatch to robot action frames ───────────────────────────────────

frames = dispatch(task_plan)
print_dispatch_plan(frames)

# ── Step 3: Set up GRUtopia environment with two H1 robots ───────────────────

headless = not has_display()

h1 = H1RobotCfg(
    name=ROBOT_NAME,
    position=(0.0, 0.0, 1.05),
    controllers=[move_by_speed_cfg, move_along_path_cfg, rotate_cfg],
    sensors=[],
)

config = Config(
    simulator=SimConfig(
        physics_dt=1 / 240,
        rendering_dt=1 / 240,
        use_fabric=False,
        headless=headless,
        webrtc=headless,
    ),
    task_configs=[
        SingleInferenceTaskCfg(
            scene_asset_path=gm.ASSET_PATH + "/scenes/demo_scenes/franka_mocap_teleop/table_scene.usd",
            scene_scale=(0.01, 0.01, 0.01),
            robots=[h1],
        ),
    ],
)

import_extensions()
env = Env(config)
obs, _ = env.reset()

print("\n[Sim] Environment ready. Starting task execution.\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_position(obs: dict) -> tuple:
    """Extract (x, y) position of the single robot from obs dict."""
    pos = obs.get("position", [0.0, 0.0, 0.0])
    return (pos[0], pos[1])


def reached(obs: dict, waypoint: tuple, threshold: float) -> bool:
    """Return True if robot is within threshold metres of the waypoint."""
    import math
    rx, ry = get_position(obs)
    wx, wy = waypoint[0], waypoint[1]
    return math.sqrt((rx - wx) ** 2 + (ry - wy) ** 2) < threshold


# ── Step 4: Flatten frames into a sequential list of (robot_label, action) ───
# Since the sim only supports one physical robot right now, we execute all
# RobotActions one after another in order, regardless of which logical robot
# (navigator/fetcher) they were assigned to. This still validates the full
# interpreter -> dispatcher -> sim pipeline; the "robot" label is logged for
# clarity and will map to physically distinct robots once available.

sequential_actions = []
for frame in frames:
    for action in frame:
        sequential_actions.append(action)

print(f"\n[Sim] Flattened into {len(sequential_actions)} sequential action(s) "
      f"(single-robot mode — see issue #59 note in script header).")

action_index   = 0
step_in_action = 0
total_steps    = 0

current_path   = sequential_actions[0].path if sequential_actions else [(0.0, 0.0, 0.0)]
current_action = {move_along_path_cfg.name: [current_path]}

while env.simulation_app.is_running():

    if action_index < len(sequential_actions):
        action = sequential_actions[action_index]
        step_in_action += 1

        final_wp = action.path[-1]
        done = reached(obs, final_wp, ARRIVAL_THRESHOLD)

        if done or step_in_action >= FRAME_TIMEOUT_STEPS:
            reason = "arrived" if done else "timeout"
            print(f"[Sim] Action {action_index+1}/{len(sequential_actions)} "
                  f"[{action.robot}:{action.task_type} -> {action.target}] "
                  f"complete ({reason}). Steps: {step_in_action}")
            action_index  += 1
            step_in_action = 0

            if action_index < len(sequential_actions):
                next_action  = sequential_actions[action_index]
                current_path = next_action.path
                current_action = {move_along_path_cfg.name: [current_path]}
                print(f"[Sim] Starting action {action_index+1}/{len(sequential_actions)} "
                      f"[{next_action.robot}:{next_action.task_type} -> {next_action.target}]")
            else:
                print("[Sim] All actions complete. Task finished!")

    obs, _, terminated, _, _ = env.step(action=current_action)
    total_steps += 1

    if total_steps % 240 == 0:
        pos = get_position(obs)
        label = (sequential_actions[action_index].robot
                 if action_index < len(sequential_actions) else "done")
        print(f"[Sim] step={total_steps:>6}  pos=({pos[0]:+.2f},{pos[1]:+.2f})  "
              f"active_robot={label}  action={action_index+1}/{len(sequential_actions)}")

    if terminated:
        print("[Sim] Episode terminated — resetting.")
        obs, _ = env.reset()

env.close()


"""
mock_task_sim.py

Identical pipeline to llm_task_sim.py but replaces the LLM interpreter
with a hardcoded mock — no model loading, no GPU needed for the LLM step.
Useful for demos and supervisor presentations where fast startup matters.

Change MOCK_INSTRUCTION at the top to switch between the four scenarios.

Run:
    python mock_task_sim.py
    (no CUDA_VISIBLE_DEVICES needed — the LLM is gone)
"""

import json
import math

from task_dispatcher import dispatch, print_dispatch_plan
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

# ── Pick one instruction to demo ──────────────────────────────────────────────
# Options:
#   "Where is the nearest bathroom?"
#   "Bring me my water bottle from the kitchen"
#   "What is around me?"
#   "Take me to the exit"

MOCK_INSTRUCTION = "Bring me my water bottle from the kitchen"

# ── Timing ────────────────────────────────────────────────────────────────────
FRAME_TIMEOUT_STEPS = 720   # ~3 seconds per action at physics_dt=1/240
ARRIVAL_THRESHOLD   = 0.5   # metres

ROBOT_NAME = "h1"

# ── Mock task plans ───────────────────────────────────────────────────────────
# These are the exact JSON structures the real LLM produces, hardcoded so the
# demo runs instantly without any model inference.

MOCK_PLANS = {
    "Where is the nearest bathroom?": {
        "instruction": "Where is the nearest bathroom?",
        "robot": "navigator",
        "task_type": "search",
        "target": "bathroom",
        "parameters": {
            "detect": "WC sign, toilet door, or bathroom entrance",
            "search_pattern": "nearest_first",
            "return_description": True,
        },
        "priority": "high",
        "explanation": (
            "Navigator searches for bathroom visual indicators "
            "and reports back to orient the blind user."
        ),
    },

    "Bring me my water bottle from the kitchen": {
        "instruction": "Bring me my water bottle from the kitchen",
        "sequence": [
            {
                "step": 1,
                "robot": "navigator",
                "task_type": "navigate",
                "target": "kitchen",
                "parameters": {"destination": "kitchen", "avoid_obstacles": True},
                "explanation": "Navigator moves to kitchen.",
            },
            {
                "step": 2,
                "robot": "navigator",
                "task_type": "search",
                "target": "water bottle",
                "parameters": {
                    "detect": "water bottle",
                    "search_pattern": "systematic",
                    "return_description": False,
                },
                "explanation": "Navigator locates the bottle visually.",
            },
            {
                "step": 3,
                "robot": "fetcher",
                "task_type": "fetch",
                "target": "water bottle",
                "parameters": {"object": "water bottle", "deliver_to": "user"},
                "explanation": "Fetcher picks up and delivers the bottle.",
            },
            {
                "step": 4,
                "robot": "fetcher",
                "task_type": "return_to_user",
                "target": "user",
                "parameters": {"carrying": "water bottle"},
                "explanation": "Fetcher returns to user with the water bottle.",
            },
        ],
        "priority": "normal",
        "overall_explanation": "Navigate to kitchen, find bottle, fetch and deliver.",
    },

    "What is around me?": {
        "instruction": "What is around me?",
        "robot": "navigator",
        "task_type": "survey",
        "target": "immediate surroundings",
        "parameters": {"focus": "full_360", "report_to_user": True},
        "priority": "normal",
        "explanation": (
            "Navigator surveys the full environment and reports "
            "landmarks and obstacles to the user."
        ),
    },

    "Take me to the exit": {
        "instruction": "Take me to the exit",
        "robot": "navigator",
        "task_type": "navigate",
        "target": "exit",
        "parameters": {"destination": "exit", "avoid_obstacles": True},
        "priority": "high",
        "explanation": "Navigator guides the user directly to the nearest exit.",
    },
}

# ── Step 1: Resolve mock task plan ────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Instruction : {MOCK_INSTRUCTION}")
print(f"{'='*60}\n")

task_plan = MOCK_PLANS.get(MOCK_INSTRUCTION)
if task_plan is None:
    print(f"[ERROR] No mock plan for: {repr(MOCK_INSTRUCTION)}")
    print(f"Available instructions:\n  " + "\n  ".join(MOCK_PLANS.keys()))
    exit(1)

print("[Mock] Task plan (no LLM — using hardcoded JSON):")
print(json.dumps(task_plan, indent=2))

# ── Step 2: Dispatch ──────────────────────────────────────────────────────────

frames = dispatch(task_plan)
print_dispatch_plan(frames)

# ── Step 3: GRUtopia environment ──────────────────────────────────────────────

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
            scene_asset_path=gm.ASSET_PATH + "/scenes/GRScenes-100/home_scenes/scenes/MWAX5JYKTKJZ2AABAAAAAEA8_usd/start_result_navigation.usd",
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
    pos = obs.get("position", [0.0, 0.0, 0.0])
    return (pos[0], pos[1])

def reached(obs: dict, waypoint: tuple, threshold: float) -> bool:
    rx, ry = get_position(obs)
    wx, wy = waypoint[0], waypoint[1]
    return math.sqrt((rx - wx) ** 2 + (ry - wy) ** 2) < threshold

# ── Step 4: Flatten frames and run sim loop ───────────────────────────────────

sequential_actions = [action for frame in frames for action in frame]

print(f"[Sim] {len(sequential_actions)} action(s) to execute:")
for i, a in enumerate(sequential_actions):
    print(f"  {i+1}. [{a.robot}] {a.task_type} → {a.target}  "
          f"({len(a.path)} waypoint(s))")
print()

action_index   = 0
step_in_action = 0
total_steps    = 0

current_path   = sequential_actions[0].path if sequential_actions else [(0.0, 0.0, 0.0)]
current_action = {move_along_path_cfg.name: [current_path]}

print(f"[Sim] Starting action 1/{len(sequential_actions)} "
      f"[{sequential_actions[0].robot}:{sequential_actions[0].task_type} "
      f"→ {sequential_actions[0].target}]")

while env.simulation_app.is_running():

    if action_index < len(sequential_actions):
        action = sequential_actions[action_index]
        step_in_action += 1

        final_wp = action.path[-1]
        done = reached(obs, final_wp, ARRIVAL_THRESHOLD)

        if done or step_in_action >= FRAME_TIMEOUT_STEPS:
            reason = "arrived" if done else "timeout"
            print(f"[Sim] Action {action_index+1}/{len(sequential_actions)} "
                  f"[{action.robot}:{action.task_type} → {action.target}] "
                  f"complete ({reason}). Steps: {step_in_action}")
            action_index  += 1
            step_in_action = 0

            if action_index < len(sequential_actions):
                nxt = sequential_actions[action_index]
                current_path   = nxt.path
                current_action = {move_along_path_cfg.name: [current_path]}
                print(f"[Sim] Starting action {action_index+1}/{len(sequential_actions)} "
                      f"[{nxt.robot}:{nxt.task_type} → {nxt.target}]")
            else:
                print("\n[Sim] ✓ All actions complete. Task finished!")
                print(f"[Sim]   Instruction : {MOCK_INSTRUCTION}")
                print(f"[Sim]   Total steps : {total_steps}")

    obs, _, terminated, _, _ = env.step(action=current_action)
    total_steps += 1

    if total_steps % 240 == 0:
        pos   = get_position(obs)
        label = (sequential_actions[action_index].robot
                 if action_index < len(sequential_actions) else "done")
        print(f"[Sim] step={total_steps:>6}  "
              f"pos=({pos[0]:+.2f}, {pos[1]:+.2f})  "
              f"robot={label}  "
              f"action={min(action_index+1, len(sequential_actions))}/{len(sequential_actions)}")

    if terminated:
        print("[Sim] Episode terminated — resetting.")
        obs, _ = env.reset()

env.close()
"""
task_dispatcher.py

Reads a parsed JSON task plan (from task_interpreter.py) and converts it into
a sequence of GRUtopia waypoint actions for the two H1 robots.

Since we don't have real perception or manipulation yet, each task type maps to
a symbolic movement pattern that visually represents what the robot would do:

  navigate      → move in a straight line toward a fixed destination waypoint
  search        → spiral/sweep pattern (simulated as a short zigzag path)
  fetch         → move to object location, pause, return toward user
  survey        → rotate in place (small circle of waypoints)
  return_to_user → move back to origin (0, 0)

Robot assignment:
  "navigator" → h1_navigator  (starts at  0,  0)
  "fetcher"   → h1_fetcher    (starts at  2,  0)
  "both"      → both robots get the same waypoints
"""

import json
from typing import Any

# ── Symbolic destination map ──────────────────────────────────────────────────
# In a real system these would come from a semantic map.
# For now, fixed waypoints in the empty scene.

DESTINATION_WAYPOINTS: dict[str, tuple] = {
    "kitchen":     (5.0,  3.0, 0.0),
    "bathroom":    (4.0, -3.0, 0.0),
    "exit":        (8.0,  0.0, 0.0),
    "elevator":    (6.0,  4.0, 0.0),
    "living room": (3.0,  2.0, 0.0),
    "default":     (4.0,  0.0, 0.0),   # fallback
}

USER_POSITION = (0.0, 0.0, 0.0)

# ── Path generators per task type ─────────────────────────────────────────────

def path_for_navigate(parameters: dict) -> list[tuple]:
    dest_name = parameters.get("destination", "default").lower()
    wp = DESTINATION_WAYPOINTS.get(dest_name, DESTINATION_WAYPOINTS["default"])
    return [wp]


def path_for_search(parameters: dict) -> list[tuple]:
    """Zigzag sweep to simulate a search pattern."""
    return [
        ( 2.0,  1.0, 0.0),
        ( 4.0, -1.0, 0.0),
        ( 6.0,  1.0, 0.0),
        ( 4.0,  0.0, 0.0),
    ]


def path_for_fetch(parameters: dict) -> list[tuple]:
    """Move toward object location then back toward user."""
    obj = parameters.get("object", "object")
    # Symbolic: all objects are "found" at (5, 2) for now
    print(f"[Dispatcher] Fetching: {obj} (symbolic location 5, 2)")
    return [
        (5.0,  2.0, 0.0),   # move to object
        (3.0,  1.0, 0.0),   # start return
        (0.0,  0.0, 0.0),   # back to user
    ]


def path_for_survey(_parameters: dict) -> list[tuple]:
    """Small circle to simulate rotating and surveying."""
    return [
        ( 0.5,  0.5, 0.0),
        ( 0.5, -0.5, 0.0),
        (-0.5, -0.5, 0.0),
        (-0.5,  0.5, 0.0),
        ( 0.0,  0.0, 0.0),
    ]


def path_for_return_to_user(_parameters: dict) -> list[tuple]:
    return [USER_POSITION]


PATH_GENERATORS = {
    "navigate":       path_for_navigate,
    "search":         path_for_search,
    "fetch":          path_for_fetch,
    "survey":         path_for_survey,
    "return_to_user": path_for_return_to_user,
}

# ── Core dispatcher ───────────────────────────────────────────────────────────

class RobotAction:
    """One unit of work for one robot: a path to follow."""
    def __init__(self, robot: str, task_type: str, target: str, path: list[tuple], explanation: str = ""):
        self.robot       = robot        # "navigator" | "fetcher"
        self.task_type   = task_type
        self.target      = target
        self.path        = path
        self.explanation = explanation

    def __repr__(self):
        return (f"RobotAction(robot={self.robot}, task={self.task_type}, "
                f"target={self.target}, steps={len(self.path)})")


def _action_for_step(step: dict) -> list[RobotAction]:
    """Convert one task plan step into RobotAction(s)."""
    robot      = step.get("robot", "navigator")
    task_type  = step.get("task_type", "navigate")
    target     = step.get("target", "unknown")
    parameters = step.get("parameters", {})
    explanation = step.get("explanation", "")

    gen  = PATH_GENERATORS.get(task_type, path_for_navigate)
    path = gen(parameters)

    if robot == "both":
        return [
            RobotAction("navigator", task_type, target, path, explanation),
            RobotAction("fetcher",   task_type, target, path, explanation),
        ]
    return [RobotAction(robot, task_type, target, path, explanation)]


def dispatch(task_plan: dict) -> list[list[RobotAction]]:
    """
    Convert a task plan dict into an ordered list of 'frames'.
    Each frame is a list of RobotActions that execute in parallel.

    Single-step plan  → one frame with one (or two) RobotActions.
    Sequence plan     → one frame per step (executed one after another).
    """
    frames: list[list[RobotAction]] = []

    if "sequence" in task_plan:
        for step in task_plan["sequence"]:
            frames.append(_action_for_step(step))
    else:
        frames.append(_action_for_step(task_plan))

    return frames


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_dispatch_plan(frames: list[list[RobotAction]]):
    print("\n[Dispatcher] Execution plan:")
    for i, frame in enumerate(frames):
        print(f"  Frame {i+1}:")
        for action in frame:
            print(f"    {action}")
            print(f"      path: {action.path}")
            if action.explanation:
                print(f"      note: {action.explanation}")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with a hardcoded plan (no LLM needed)
    sample_plan = {
        "instruction": "Bring me my water bottle from the kitchen",
        "sequence": [
            {
                "step": 1,
                "robot": "navigator",
                "task_type": "navigate",
                "target": "kitchen",
                "parameters": {"destination": "kitchen", "avoid_obstacles": True},
                "explanation": "Navigator moves to kitchen."
            },
            {
                "step": 2,
                "robot": "navigator",
                "task_type": "search",
                "target": "water bottle",
                "parameters": {"detect": "water bottle", "search_pattern": "systematic"},
                "explanation": "Navigator searches for the bottle."
            },
            {
                "step": 3,
                "robot": "fetcher",
                "task_type": "fetch",
                "target": "water bottle",
                "parameters": {"object": "water bottle", "deliver_to": "user"},
                "explanation": "Fetcher picks up and delivers."
            },
        ],
        "priority": "normal",
        "overall_explanation": "Navigate, find, fetch."
    }

    frames = dispatch(sample_plan)
    print_dispatch_plan(frames)
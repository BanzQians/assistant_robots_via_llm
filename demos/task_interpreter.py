"""
task_interpreter.py

Loads a local GGUF model via llama-cpp-python and converts a natural language
instruction into a structured JSON task plan.

Can be run standalone to test outputs before wiring into the sim:
    python task_interpreter.py
"""

import json
import re
from llama_cpp import Llama

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = "/cvhci/temp/isandan/models/gemma-3-4b-it-Q4_K_M.gguf"

# Mock instructions to test — swap or extend as needed
MOCK_INSTRUCTIONS = [
    "Where is the nearest bathroom?",
    "Bring me my water bottle from the kitchen",
    "What is around me?",
    "Take me to the exit",
]

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a robot task interpreter assistant for a blind user navigation and assistance system.

You receive natural language instructions from a blind user and convert them into structured JSON task plans for a two-robot system.

## The Robot System

Navigator Robot: autonomous movement, visual perception, sign/object detection, room exploration.
Fetcher Robot: object manipulation, grasping, carrying, picking up and delivering items.

## Task Types (fixed set)
- navigate    : Move to a known or described location
- search      : Explore and visually detect a target
- fetch       : Pick up and bring back a physical object
- survey      : Observe and describe the immediate surroundings
- return_to_user : Robot returns to the user's position

## Output Format

For simple single-step tasks output ONLY this JSON:
{"instruction":"...","robot":"navigator|fetcher|both","task_type":"navigate|search|fetch|survey|return_to_user","target":"...","parameters":{"key":"value"},"priority":"high|normal","explanation":"..."}

For multi-step tasks output ONLY this JSON:
{"instruction":"...","sequence":[{"step":1,"robot":"navigator|fetcher","task_type":"...","target":"...","parameters":{},"explanation":"..."}],"priority":"high|normal","overall_explanation":"..."}

## Rules
1. Output ONLY the JSON object. No prose, no markdown, no code fences.
2. Use the most specific task_type that fits.
3. Mark priority "high" for urgent needs (bathroom, exit, emergency).
4. If ambiguous, make a reasonable assumption and note it in explanation.

## Examples

User: Where is the nearest bathroom?
{"instruction":"Where is the nearest bathroom?","robot":"navigator","task_type":"search","target":"bathroom","parameters":{"detect":"WC sign, toilet door, or bathroom entrance","search_pattern":"nearest_first","return_description":true},"priority":"high","explanation":"Navigator searches for bathroom visual indicators and reports back to orient the blind user."}

User: What is around me?
{"instruction":"What is around me?","robot":"navigator","task_type":"survey","target":"immediate surroundings","parameters":{"focus":"full_360","report_to_user":true},"priority":"normal","explanation":"Navigator surveys the full environment and reports landmarks and obstacles to the user."}

User: Bring me my water bottle from the kitchen
{"instruction":"Bring me my water bottle from the kitchen","sequence":[{"step":1,"robot":"navigator","task_type":"navigate","target":"kitchen","parameters":{"destination":"kitchen","avoid_obstacles":true},"explanation":"Navigator moves to kitchen."},{"step":2,"robot":"navigator","task_type":"search","target":"water bottle","parameters":{"detect":"water bottle","search_pattern":"systematic","return_description":false},"explanation":"Navigator locates the bottle visually."},{"step":3,"robot":"fetcher","task_type":"fetch","target":"water bottle","parameters":{"object":"water bottle","deliver_to":"user"},"explanation":"Fetcher picks up and delivers the bottle."}],"priority":"normal","overall_explanation":"Navigate to kitchen, find bottle, fetch and deliver to user."}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_json(text: str) -> dict | None:
    """
    Extract and parse the first JSON object found in the model output.
    Handles cases where the model wraps output in markdown fences or adds prose.
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try parsing the whole string first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def interpret(llm: Llama, instruction: str) -> dict | None:
    """
    Send a user instruction to the LLM and return the parsed task plan dict.
    Returns None if the model output cannot be parsed as valid JSON.
    """
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"User: {instruction}\n"
        f"JSON:"
    )

    output = llm(
        prompt,
        max_tokens=512,
        stop=["User:", "##"],
        echo=False,
    )

    raw = output["choices"][0]["text"].strip()
    print(f"\n[Interpreter] Raw output:\n{raw}")

    task_plan = extract_json(raw)
    if task_plan is None:
        print("[Interpreter] WARNING: could not parse JSON from model output.")
    return task_plan


# ── Main (standalone test) ────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[Interpreter] Loading model from {MODEL_PATH} ...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )
    print("[Interpreter] Model loaded.\n")

    for instruction in MOCK_INSTRUCTIONS:
        print("=" * 60)
        print(f"Instruction: {instruction}")
        plan = interpret(llm, instruction)
        if plan:
            print(f"\n[Interpreter] Parsed task plan:")
            print(json.dumps(plan, indent=2))
        else:
            print("[Interpreter] Failed to produce a valid task plan.")
"""B2Z1 完整抓取 demo：走路接近 → IK 抓取 → 搬运 → 放下

FSM 状态：
  STABILIZE   → 静止站稳，策略稳定
  APPROACH    → 走路接近物体
  ALIGN       → 对准物体朝向，微调距离
  GRASP       → IK 控制臂伸向物体，夹爪闭合
  LIFT        → 抬起物体（臂向上移动）
  CARRY       → 走路搬运到目标位置
  RELEASE     → 放下物体，夹爪张开，臂归位

用法：
    python b2z1_grasp_demo.py                     # 用 box（默认）
    python b2z1_grasp_demo.py --object chair      # 用椅子
    python b2z1_grasp_demo.py --headless          # 无界面运行
"""

import argparse
import math

# Import pinocchio before Isaac Sim starts so its (cxx11-ABI) libassimp claims the
# libassimp.so.5 soname first; otherwise Isaac's older assimp loads first and the
# Z1 IK solver fails with "undefined symbol ... Assimp::IOSystem::CurrentDirectory".
import pinocchio  # noqa: F401

import numpy as np
from scipy.spatial.transform import Rotation

from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia.macros import gm
from internutopia_extension import import_extensions
from internutopia_extension.configs.objects import UsdObjCfg
from internutopia_extension.configs.robots.b2z1 import B2Z1RobotCfg
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg
from internutopia_extension.controllers.z1_ik_solver import Z1IKSolver

# ── Object asset paths ─────────────────────────────────────────────────────────
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
_ASSETS = os.path.join(_REPO_ROOT, 'local_assets', 'objects', 'alore')

OBJECT_CFGS = {
    'box': {
        'usd':        os.path.join(_ASSETS, 'object_7/box.usd'),
        'spawn_pos':  (1.5, 0.0, 0.05),
        'grasp_cfg':  (0.25, 0.25, 80.0),   # dx, dz, pitch_deg
        'plan_dist':  0.83,                  # how close robot walks to object
        'com_offset': (0.0, 0.0),
    },
    'chair': {
        'usd':        os.path.join(_ASSETS, 'object_1/model_office_chair_3_v1.usd'),
        'spawn_pos':  (1.8, 0.0, 0.0),
        'grasp_cfg':  (0.45, 0.80, 60.0),
        'plan_dist':  0.80,
        'com_offset': (0.0, 0.0),
    },
}

# ── Timing (in seconds) ────────────────────────────────────────────────────────
HZ = 200
STABILIZE_SECS = 2.0
GRASP_APPROACH_SECS = 4.0   # max time to extend arm
GRIPPER_CLOSE_SECS  = 3.0
CARRY_SECS          = 5.0
RELEASE_SECS        = 2.0

# Arm home pose (folded on back)
ARM_HOME = np.array([0.0, 1.48, -0.63, -0.84, 0.0, 1.57, -1.5])  # -1.5 = open gripper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--object', choices=['box', 'chair'], default='box')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--target-x', type=float, default=-1.5, help='Target x to place object')
    p.add_argument('--target-y', type=float, default=0.0,  help='Target y to place object')
    return p.parse_args()


def get_object_world_pose(prim_path: str):
    """Query object position and orientation from Isaac Sim stage."""
    from omni.isaac.core.prims import RigidPrim
    prim = RigidPrim(prim_path=prim_path, name=prim_path.lstrip('/'))
    pos, quat_wxyz = prim.get_world_pose()   # quat: (w,x,y,z)
    yaw = Rotation.from_quat(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    ).as_euler('xyz')[2]
    return np.array(pos), float(yaw)


def angle_diff(a, b):
    """Signed angle difference (a - b) in [-pi, pi]."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


def robot_yaw_from_obs(obs) -> float:
    quat_wxyz = obs['orientation']   # (w,x,y,z)
    return float(Rotation.from_quat(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    ).as_euler('xyz')[2])


def main():
    args = parse_args()
    headless = args.headless or not has_display()
    obj_cfg = OBJECT_CFGS[args.object]

    config = Config(
        simulator=SimConfig(
            physics_dt=1 / HZ,
            rendering_dt=1 / 50,
            use_fabric=False,
            headless=headless,
            webrtc=headless,
        ),
        task_configs=[
            SingleInferenceTaskCfg(
                scene_asset_path=gm.ASSET_PATH + '/scenes/empty.usd',
                robots=[B2Z1RobotCfg(position=(0.0, 0.0, 0.55))],
                objects=[
                    UsdObjCfg(
                        name='target_object',
                        prim_path='/target_object',
                        usd_path=obj_cfg['usd'],
                        position=obj_cfg['spawn_pos'],
                    )
                ],
            )
        ],
    )

    import_extensions()
    env = Env(config)
    obs, _ = env.reset()

    ik_solver = Z1IKSolver()

    print(f'=== B2Z1 Grasp Demo  object={args.object} ===')
    print(f'  Object spawned at : {obj_cfg["spawn_pos"]}')
    print(f'  Target placement  : ({args.target_x:.1f}, {args.target_y:.1f})')

    dx_grasp, dz_grasp, pitch_grasp = obj_cfg['grasp_cfg']
    plan_dist = obj_cfg['plan_dist']
    target_place = np.array([args.target_x, args.target_y, obj_cfg['spawn_pos'][2]])

    # FSM state
    state = 'STABILIZE'
    t0 = 0.0
    step = 0
    gripper_angle = -1.5    # open
    last_q_arm = ARM_HOME.copy()
    phase_log: set = set()

    try:
        while env.simulation_app.is_running():
            step += 1
            t = step / HZ

            robot_pos = np.array(obs['position'])
            robot_yaw = robot_yaw_from_obs(obs)

            # ── Query object pose from scene ──────────────────────────────────
            try:
                obj_pos, obj_yaw = get_object_world_pose('/target_object')
            except Exception:
                obj_pos = np.array(obj_cfg['spawn_pos'])
                obj_yaw = 0.0

            loco_cmd = [0.0, 0.0, 0.0]   # forward, lateral, rotation
            arm_cmd   = last_q_arm.copy() # default: hold last pose

            # ──────────────────────────────────────────────────────────────────
            if state == 'STABILIZE':
                if state not in phase_log:
                    print(f'[{t:.1f}s] STABILIZE: policy settling...')
                    phase_log.add(state); t0 = t
                arm_cmd = ARM_HOME.copy()
                if t - t0 > STABILIZE_SECS:
                    state = 'APPROACH'; t0 = t

            elif state == 'APPROACH':
                if state not in phase_log:
                    print(f'[{t:.1f}s] APPROACH: walking toward object...')
                    phase_log.add(state)

                dx = obj_pos[0] - robot_pos[0]
                dy = obj_pos[1] - robot_pos[1]
                dist = math.hypot(dx, dy)
                target_yaw = math.atan2(dy, dx)
                yaw_err = angle_diff(target_yaw, robot_yaw)

                arm_cmd = ARM_HOME.copy()   # keep arm folded while walking

                if dist < plan_dist + 0.1:
                    print(f'[{t:.1f}s] APPROACH done (dist={dist:.2f}m). → ALIGN')
                    state = 'ALIGN'; t0 = t
                else:
                    vx = 0.4 if abs(yaw_err) < math.radians(20) else 0.0
                    wz = np.clip(2.0 * yaw_err, -0.6, 0.6)
                    loco_cmd = [vx, 0.0, wz]

            elif state == 'ALIGN':
                if state not in phase_log:
                    print(f'[{t:.1f}s] ALIGN: face object, fine distance...')
                    phase_log.add(state)

                dx = obj_pos[0] - robot_pos[0]
                dy = obj_pos[1] - robot_pos[1]
                dist = math.hypot(dx, dy)
                target_yaw = math.atan2(dy, dx)
                yaw_err = angle_diff(target_yaw, robot_yaw)
                dist_err = dist - plan_dist

                wz = np.clip(2.0 * yaw_err, -0.4, 0.4)
                vx = np.clip(1.5 * dist_err, -0.15, 0.15)
                loco_cmd = [vx, 0.0, wz]
                arm_cmd = ARM_HOME.copy()

                if abs(yaw_err) < math.radians(8) and abs(dist_err) < 0.05:
                    print(f'[{t:.1f}s] ALIGN done. → GRASP')
                    state = 'GRASP'; t0 = t
                    gripper_angle = -1.5   # ensure open

            elif state == 'GRASP':
                if state not in phase_log:
                    print(f'[{t:.1f}s] GRASP: IK arm reaching...')
                    phase_log.add(state)

                loco_cmd = [0.0, 0.0, 0.0]   # hold still
                elapsed = t - t0

                # Compute grasp EE target in world frame
                grasp_pos_world, grasp_rot_world = Z1IKSolver.grasp_pose_in_world(
                    obj_pos, obj_yaw, dx_grasp, dz_grasp, pitch_grasp
                )

                # Solve IK via arm_ik controller
                # Build action: [x,y,z, qx,qy,qz,qw, gripper]
                quat_xyzw = Rotation.from_matrix(grasp_rot_world).as_quat()
                # Gradually close gripper after arm has had time to reach
                if elapsed > GRASP_APPROACH_SECS * 0.6:
                    close_frac = np.clip(
                        (elapsed - GRASP_APPROACH_SECS * 0.6) / GRIPPER_CLOSE_SECS,
                        0.0, 1.0
                    )
                    gripper_angle = -1.5 + close_frac * 1.4   # -1.5 → -0.1
                    gripper_angle = np.clip(gripper_angle, -1.5, -0.1)

                arm_cmd_ik = np.concatenate([grasp_pos_world, quat_xyzw, [gripper_angle]])

                action = {
                    'move_by_speed': loco_cmd,
                    'arm_ik': arm_cmd_ik,
                }
                obs, _, terminated, _, _ = env.step(action=action)
                last_q_arm = obs.get('controllers', {}).get('arm_ik', {}).get('q', last_q_arm)
                if isinstance(last_q_arm, list):
                    last_q_arm = np.array(last_q_arm)

                if elapsed > GRASP_APPROACH_SECS + GRIPPER_CLOSE_SECS:
                    print(f'[{t:.1f}s] GRASP done. → CARRY')
                    state = 'CARRY'; t0 = t
                continue  # skip default step at bottom

            elif state == 'CARRY':
                if state not in phase_log:
                    print(f'[{t:.1f}s] CARRY: walking to target ({args.target_x:.1f},{args.target_y:.1f})...')
                    phase_log.add(state)

                dx = target_place[0] - robot_pos[0]
                dy = target_place[1] - robot_pos[1]
                dist = math.hypot(dx, dy)
                target_yaw = math.atan2(dy, dx)
                yaw_err = angle_diff(target_yaw, robot_yaw)

                # Keep arm in grasp pose while carrying
                grasp_pos_world, grasp_rot_world = Z1IKSolver.grasp_pose_in_world(
                    obj_pos, obj_yaw, dx_grasp, dz_grasp, pitch_grasp
                )
                quat_xyzw = Rotation.from_matrix(grasp_rot_world).as_quat()
                arm_cmd_ik = np.concatenate([grasp_pos_world, quat_xyzw, [gripper_angle]])

                vx = 0.35 if abs(yaw_err) < math.radians(20) else 0.0
                wz = np.clip(2.0 * yaw_err, -0.5, 0.5)
                loco_cmd = [vx, 0.0, wz]

                action = {
                    'move_by_speed': loco_cmd,
                    'arm_ik': arm_cmd_ik,
                }
                obs, _, terminated, _, _ = env.step(action=action)

                if dist < 0.3 or t - t0 > CARRY_SECS:
                    print(f'[{t:.1f}s] CARRY done. → RELEASE')
                    state = 'RELEASE'; t0 = t
                continue

            elif state == 'RELEASE':
                if state not in phase_log:
                    print(f'[{t:.1f}s] RELEASE: opening gripper, arm home...')
                    phase_log.add(state)

                loco_cmd = [0.0, 0.0, 0.0]
                elapsed = t - t0
                # Open gripper first, then move arm home
                open_frac = np.clip(elapsed / 1.0, 0.0, 1.0)
                gripper_angle = -0.1 + open_frac * (-1.4)   # -0.1 → -1.5
                arm_cmd = ARM_HOME.copy()
                arm_cmd[6] = gripper_angle

                if elapsed > RELEASE_SECS:
                    print(f'[{t:.1f}s] RELEASE done. All tasks complete!')
                    break

            # ── Default step (non-GRASP/CARRY phases) ─────────────────────────
            action = {
                'move_by_speed': loco_cmd,
                'arm_control':   [arm_cmd],
            }
            obs, _, terminated, _, _ = env.step(action=action)
            last_q_arm = np.array(arm_cmd)

            if step % HZ == 0:
                print(f'  t={t:5.1f}s  state={state:12s}  '
                      f'robot=({robot_pos[0]:.2f},{robot_pos[1]:.2f})  '
                      f'obj=({obj_pos[0]:.2f},{obj_pos[1]:.2f})')

            if terminated:
                break

    finally:
        print('\nDemo finished.')
        env.close()


if __name__ == '__main__':
    main()

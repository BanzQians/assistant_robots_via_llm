"""B2Z1 JSON 驱动的多任务抓取搬运运行器。

从外部 JSON 文件读取物体列表和任务序列，机器人依次：
  1. 走到物体附近 (APPROACH)
  2. 对准物体 (ALIGN)
  3. 判断物体类别 → 选择对应抓取参数 (IDENTIFY)
  4. IK 控制机械臂抓取 (GRASP)
  5. 搬运到目标点 (CARRY)
  6. 放下、机械臂归位 (RELEASE)
  7. 处理下一个任务

用法：
    python b2z1_task_runner.py                              # 用默认示例 JSON
    python b2z1_task_runner.py --tasks my_tasks.json        # 自定义任务文件
    python b2z1_task_runner.py --headless                   # 无界面

JSON 格式见 b2z1_tasks_example.json。
"""

import argparse
import json
import math
import os

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

from b2z1_object_library import get_object_params

# ── 时间常数 ────────────────────────────────────────────────────────────────────
HZ = 200
STABILIZE_SECS      = 2.0
GRASP_REACH_SECS    = 4.0
GRIPPER_CLOSE_SECS  = 3.0
CARRY_TIMEOUT_SECS  = 8.0
RELEASE_SECS        = 2.5

# 机械臂折叠（home）姿态：joint1-6 + gripper，-1.5 = 张开
ARM_HOME = np.array([0.0, 1.48, -0.63, -0.84, 0.0, 1.57, -1.5])


# ── 辅助函数 ────────────────────────────────────────────────────────────────────
def angle_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def quat_wxyz_to_yaw(quat_wxyz):
    return float(Rotation.from_quat(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    ).as_euler('xyz')[2])


def get_object_world_pose(prim_path: str):
    """从 Isaac Sim stage 查询物体位置和 yaw。"""
    from omni.isaac.core.prims import RigidPrim
    prim = RigidPrim(prim_path=prim_path, name=prim_path.lstrip('/').replace('/', '_'))
    pos, quat_wxyz = prim.get_world_pose()
    return np.array(pos), quat_wxyz_to_yaw(quat_wxyz)


def parse_args():
    p = argparse.ArgumentParser(description='B2Z1 JSON-driven multi-task grasp runner.')
    default_json = os.path.join(os.path.dirname(__file__), 'b2z1_tasks_example.json')
    p.add_argument('--tasks', default=default_json, help='任务 JSON 文件路径')
    p.add_argument('--headless', action='store_true')
    return p.parse_args()


def load_task_spec(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        spec = json.load(f)
    # 校验 + 给每个 object 补上抓取参数
    obj_by_name = {}
    for obj in spec['objects']:
        params = get_object_params(obj['category'])   # 未知类别会在这里报错
        obj['_params'] = params
        obj_by_name[obj['name']] = obj
    spec['_obj_by_name'] = obj_by_name
    # 校验 task 引用的 object 存在
    for task in spec['tasks']:
        if task['object'] not in obj_by_name:
            raise ValueError(f'任务引用了未定义的物体: {task["object"]}')
    return spec


# ── 主流程 ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    headless = args.headless or not has_display()

    spec = load_task_spec(args.tasks)
    robot_start = tuple(spec.get('robot_start', [0.0, 0.0, 0.55]))

    print('=== B2Z1 多任务运行器 ===')
    print(f'  任务文件 : {args.tasks}')
    print(f'  物体数量 : {len(spec["objects"])}')
    print(f'  任务数量 : {len(spec["tasks"])}')
    for i, task in enumerate(spec['tasks']):
        obj = spec['_obj_by_name'][task['object']]
        print(f'    任务{i+1}: 抓取 "{task["object"]}" (类别={obj["category"]}) → 放到 {task["target"]}')

    # 构建场景物体
    object_cfgs = []
    for obj in spec['objects']:
        object_cfgs.append(UsdObjCfg(
            name=obj['name'],
            prim_path='/' + obj['name'],
            usd_path=obj['_params']['usd'],
            position=tuple(obj['position']),
        ))

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
                robots=[B2Z1RobotCfg(position=robot_start)],
                objects=object_cfgs,
            )
        ],
    )

    import_extensions()
    env = Env(config)
    obs, _ = env.reset()

    # ── FSM 状态 ──────────────────────────────────────────────────────────────
    task_idx = 0
    state = 'STABILIZE'
    t0 = 0.0
    step = 0
    gripper_angle = -1.5
    last_arm = ARM_HOME.copy()
    phase_log = set()

    # 当前任务的缓存
    cur_obj = None
    cur_params = None
    cur_target = None

    def begin_task(idx):
        nonlocal cur_obj, cur_params, cur_target, phase_log
        task = spec['tasks'][idx]
        cur_obj = spec['_obj_by_name'][task['object']]
        cur_params = cur_obj['_params']
        cur_target = np.array(task['target'] + [cur_obj['position'][2]]) \
            if len(task['target']) == 2 else np.array(task['target'])
        phase_log = set()
        print(f'\n──── 开始任务 {idx+1}/{len(spec["tasks"])}: {task["object"]} ────')

    begin_task(0)

    try:
        while env.simulation_app.is_running():
            step += 1
            t = step / HZ

            robot_pos = np.array(obs['position'])
            robot_yaw = quat_wxyz_to_yaw(obs['orientation'])

            # 查询当前任务物体位姿
            try:
                obj_pos, obj_yaw = get_object_world_pose('/' + cur_obj['name'])
            except Exception:
                obj_pos = np.array(cur_obj['position']); obj_yaw = 0.0

            dx_g, dz_g, pitch_g = cur_params['grasp_cfg']
            plan_dist = cur_params['plan_dist']
            grip_close = cur_params['grip_close']

            loco_cmd = [0.0, 0.0, 0.0]
            arm_action_ik = None       # 若非 None，则用 arm_ik 控制器
            arm_action_joint = last_arm.copy()  # 否则用 arm_control 直接关节

            # ── STABILIZE ──────────────────────────────────────────────────
            if state == 'STABILIZE':
                if state not in phase_log:
                    print(f'[{t:.1f}s] STABILIZE: 策略稳定中...'); phase_log.add(state); t0 = t
                arm_action_joint = ARM_HOME.copy()
                if t - t0 > STABILIZE_SECS:
                    state = 'APPROACH'

            # ── APPROACH ───────────────────────────────────────────────────
            elif state == 'APPROACH':
                if state not in phase_log:
                    print(f'[{t:.1f}s] APPROACH: 走向 "{cur_obj["name"]}"...'); phase_log.add(state)
                dx = obj_pos[0] - robot_pos[0]; dy = obj_pos[1] - robot_pos[1]
                dist = math.hypot(dx, dy)
                yaw_err = angle_diff(math.atan2(dy, dx), robot_yaw)
                arm_action_joint = ARM_HOME.copy()
                if dist < plan_dist + 0.15:
                    state = 'ALIGN'
                else:
                    vx = 0.4 if abs(yaw_err) < math.radians(20) else 0.0
                    loco_cmd = [vx, 0.0, float(np.clip(2.0 * yaw_err, -0.6, 0.6))]

            # ── ALIGN ──────────────────────────────────────────────────────
            elif state == 'ALIGN':
                if state not in phase_log:
                    print(f'[{t:.1f}s] ALIGN: 对准物体...'); phase_log.add(state)
                dx = obj_pos[0] - robot_pos[0]; dy = obj_pos[1] - robot_pos[1]
                dist = math.hypot(dx, dy)
                yaw_err = angle_diff(math.atan2(dy, dx), robot_yaw)
                dist_err = dist - plan_dist
                loco_cmd = [
                    float(np.clip(1.5 * dist_err, -0.15, 0.15)),
                    0.0,
                    float(np.clip(2.0 * yaw_err, -0.4, 0.4)),
                ]
                arm_action_joint = ARM_HOME.copy()
                if abs(yaw_err) < math.radians(8) and abs(dist_err) < 0.06:
                    state = 'IDENTIFY'

            # ── IDENTIFY（判断物体类别 → 选择抓取策略）─────────────────────
            elif state == 'IDENTIFY':
                # 此处类别来自 JSON 规格；将来可替换为相机感知 / 分类网络
                print(f'[{t:.1f}s] IDENTIFY: 识别物体类别 = "{cur_obj["category"]}"')
                print(f'           → 抓取参数 dx={dx_g} dz={dz_g} pitch={pitch_g}° '
                      f'夹爪闭合角={grip_close}')
                gripper_angle = -1.5
                state = 'GRASP'; t0 = t

            # ── GRASP（IK 伸臂 + 闭合夹爪）─────────────────────────────────
            elif state == 'GRASP':
                if state not in phase_log:
                    print(f'[{t:.1f}s] GRASP: IK 伸臂抓取...'); phase_log.add(state)
                elapsed = t - t0
                grasp_pos_w, grasp_rot_w = Z1IKSolver.grasp_pose_in_world(
                    obj_pos, obj_yaw, dx_g, dz_g, pitch_g)
                quat_xyzw = Rotation.from_matrix(grasp_rot_w).as_quat()
                # 臂先伸到位，再逐渐从张开(-1.5)闭合到 grip_close
                if elapsed > GRASP_REACH_SECS * 0.6:
                    frac = np.clip((elapsed - GRASP_REACH_SECS * 0.6) / GRIPPER_CLOSE_SECS, 0.0, 1.0)
                    gripper_angle = -1.5 + frac * (grip_close - (-1.5))
                arm_action_ik = np.concatenate([grasp_pos_w, quat_xyzw, [gripper_angle]])
                if elapsed > GRASP_REACH_SECS + GRIPPER_CLOSE_SECS:
                    state = 'CARRY'; t0 = t

            # ── CARRY（搬运到目标点）───────────────────────────────────────
            elif state == 'CARRY':
                if state not in phase_log:
                    print(f'[{t:.1f}s] CARRY: 搬运到 {cur_target[:2]}...'); phase_log.add(state)
                dx = cur_target[0] - robot_pos[0]; dy = cur_target[1] - robot_pos[1]
                dist = math.hypot(dx, dy)
                yaw_err = angle_diff(math.atan2(dy, dx), robot_yaw)
                # 搬运时保持抓取姿态
                grasp_pos_w, grasp_rot_w = Z1IKSolver.grasp_pose_in_world(
                    obj_pos, obj_yaw, dx_g, dz_g, pitch_g)
                quat_xyzw = Rotation.from_matrix(grasp_rot_w).as_quat()
                arm_action_ik = np.concatenate([grasp_pos_w, quat_xyzw, [gripper_angle]])
                vx = 0.35 if abs(yaw_err) < math.radians(20) else 0.0
                loco_cmd = [vx, 0.0, float(np.clip(2.0 * yaw_err, -0.5, 0.5))]
                if dist < 0.35 or t - t0 > CARRY_TIMEOUT_SECS:
                    state = 'RELEASE'; t0 = t

            # ── RELEASE（放下 + 归位）──────────────────────────────────────
            elif state == 'RELEASE':
                if state not in phase_log:
                    print(f'[{t:.1f}s] RELEASE: 放下物体，机械臂归位...'); phase_log.add(state)
                elapsed = t - t0
                frac = np.clip(elapsed / 1.0, 0.0, 1.0)
                gripper_angle = grip_close + frac * (-1.5 - grip_close)
                arm_action_joint = ARM_HOME.copy()
                arm_action_joint[6] = gripper_angle
                if elapsed > RELEASE_SECS:
                    # 进入下一个任务
                    task_idx += 1
                    if task_idx >= len(spec['tasks']):
                        print(f'\n[{t:.1f}s] ✓ 所有任务完成！')
                        break
                    begin_task(task_idx)
                    state = 'APPROACH'
                    last_arm = ARM_HOME.copy()

            # ── 组装 action 并执行单步 ─────────────────────────────────────
            if arm_action_ik is not None:
                action = {'move_by_speed': loco_cmd, 'arm_ik': arm_action_ik}
            else:
                action = {'move_by_speed': loco_cmd, 'arm_control': [arm_action_joint]}
                last_arm = np.array(arm_action_joint)

            obs, _, terminated, _, _ = env.step(action=action)

            # 从 arm_ik 控制器回读关节角，供 RELEASE 阶段平滑过渡
            if arm_action_ik is not None:
                q = obs.get('controllers', {}).get('arm_ik', {}).get('q')
                if q is not None:
                    last_arm = np.array(q)

            if step % HZ == 0:
                print(f'  t={t:5.1f}s  task={task_idx+1}  state={state:10s}  '
                      f'robot=({robot_pos[0]:.2f},{robot_pos[1]:.2f},yaw={math.degrees(robot_yaw):.0f}°)')

            if terminated:
                print(f'[{t:.1f}s] 仿真 terminated。')
                break

    finally:
        print('\n运行结束。')
        env.close()


if __name__ == '__main__':
    main()

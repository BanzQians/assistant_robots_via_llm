"""Demo: B2Z1 locomotion using InternUtopia + ALORE low-level policy.

Usage:
    python b2z1_locomotion.py                          # walk forward slowly
    python b2z1_locomotion.py --forward-speed 0.3      # faster
    python b2z1_locomotion.py --scene mini_home        # small generated home scene
    python b2z1_locomotion.py --scene home             # walk in a GRScenes home
    python b2z1_locomotion.py --scene commercial       # walk in a GRScenes commercial scene
    python b2z1_locomotion.py --headless --steps 500   # headless for 500 steps
"""

import argparse
import os

import numpy as np

from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia.macros import gm
from internutopia_extension import import_extensions
from internutopia_extension.configs.controllers import JointControllerCfg
from internutopia_extension.configs.robots.b2z1 import (
    B2Z1RobotCfg,
    B2Z1_LEG_JOINTS,
    arm_controller_cfg,
    move_by_speed_cfg,
)
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg

# B2 arm home pose (matches ALORE default_joint_pos_robot arm part)
ARM_HOME = np.array([0.0, 1.48, -0.63, -0.84, 0.0, 1.57, -1.2])
BODY_HOLD = np.array([
    0.1, -0.1, 0.1, -0.1,
    0.8, 0.8, 0.8, 0.8,
    ARM_HOME[0],
    -1.5, -1.5, -1.5, -1.5,
    ARM_HOME[1], ARM_HOME[2], ARM_HOME[3], ARM_HOME[4], ARM_HOME[5], ARM_HOME[6],
])
body_hold_controller_cfg = JointControllerCfg(name='body_hold', joint_names=B2Z1_LEG_JOINTS)

SCENE_PRESETS = {
    'empty': 'scenes/empty.usd',
    'mini_home': 'scenes/empty.usd',
    'home': (
        'scenes/GRScenes-100/home_scenes/scenes/'
        'MWBGLKQKTKJZ2AABAAAAACA8_usd/start_result_navigation.usd'
    ),
    'commercial': (
        'scenes/GRScenes-100/commercial_scenes/scenes/'
        'MV4AFHQKTKJZ2AABAAAAAEA8_usd/start_result_navigation.usd'
    ),
}

GENERATED_SCENES = {'mini_home'}


def quat_wxyz_to_yaw(quat):
    w, x, y, z = np.array(quat, dtype=float)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def quat_wxyz_to_rpy(quat):
    w, x, y, z = np.array(quat, dtype=float)
    roll = float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    pitch = float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return roll, pitch, yaw


def wrap_to_pi(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def parse_vec3(text, arg_name):
    try:
        values = tuple(float(item.strip()) for item in text.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'{arg_name} must be three comma-separated numbers.') from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f'{arg_name} must have exactly three values.')
    return values


def apply_gui_view(robot_position, auto_camera=True):
    """Point the visible viewport at the robot after the USD stage is populated."""
    stage = None
    try:
        import omni.usd
        from pxr import UsdLux

        stage = omni.usd.get_context().get_stage()
        if stage is not None and not stage.GetPrimAtPath('/World/DefaultDomeLight'):
            dome_light = UsdLux.DomeLight.Define(stage, '/World/DefaultDomeLight')
            dome_light.CreateIntensityAttr(800.0)

        selection = omni.usd.get_context().get_selection()
        selection.set_selected_prim_paths(['/World/env_0/robots/b2z1/base'], True)
        print('[B2Z1 demo] GUI hint: selected /World/env_0/robots/b2z1/base. Click the viewport and press F to frame it.', flush=True)
    except Exception as exc:
        print(f'[B2Z1 demo] Failed to select/light stage: {exc}', flush=True)

    if not auto_camera or stage is None:
        return

    try:
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, UsdGeom

        try:
            from isaacsim.core.utils.viewports import set_camera_view
        except ModuleNotFoundError:
            from omni.isaac.core.utils.viewports import set_camera_view

        target = np.array(robot_position, dtype=float)
        target[2] = max(target[2], 0.6)
        eye = target + np.array([3.0, -4.0, 2.2])

        camera_path = '/World/B2Z1DebugCamera'
        camera = UsdGeom.Camera.Define(stage, camera_path)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        camera.CreateFocalLengthAttr(18.0)

        viewport = get_active_viewport()
        if viewport is None:
            print('[B2Z1 demo] Failed to get active viewport for camera switch.', flush=True)
            return

        set_camera_view(eye=eye, target=target, camera_prim_path=camera_path, viewport_api=viewport)
        try:
            viewport.camera_path = camera_path
        except TypeError:
            from pxr import Sdf

            viewport.camera_path = Sdf.Path(camera_path)
        print(
            '[B2Z1 demo] GUI camera: '
            f'{camera_path} eye={np.round(eye, 3).tolist()} target={np.round(target, 3).tolist()}',
            flush=True,
        )
    except Exception as exc:
        print(f'[B2Z1 demo] Failed to set viewport camera: {exc}', flush=True)


def add_mini_home_scene():
    """Create a small visible room with simple collision geometry."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root_path = '/World/env_0/scene/mini_home'
    if stage is None or stage.GetPrimAtPath(root_path).IsValid():
        return

    UsdGeom.Xform.Define(stage, root_path)

    def box(name, center, size, color):
        path = f'{root_path}/{name}'
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])

        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))

        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        return cube

    # Floor and three walls: open in +X so the dog has a visible route forward.
    box('ground_floor', (0.0, 0.0, -0.03), (8.0, 6.0, 0.06), (0.55, 0.55, 0.55))
    box('wall_back', (-4.0, 0.0, 1.0), (0.08, 6.0, 2.0), (0.78, 0.78, 0.72))
    box('wall_left', (0.0, 3.0, 1.0), (8.0, 0.08, 2.0), (0.78, 0.78, 0.72))
    box('wall_right', (0.0, -3.0, 1.0), (8.0, 0.08, 2.0), (0.78, 0.78, 0.72))

    # Simple furniture kept off the center path.
    box('sofa', (-1.8, 1.8, 0.35), (1.4, 0.55, 0.7), (0.1, 0.28, 0.75))
    box('coffee_table', (1.4, 1.1, 0.25), (1.0, 0.6, 0.5), (0.48, 0.25, 0.12))
    box('shelf', (2.3, -2.2, 0.85), (0.7, 0.45, 1.7), (0.42, 0.24, 0.1))
    box('counter', (3.0, 1.9, 0.45), (1.3, 0.55, 0.9), (0.65, 0.65, 0.6))

    print('[B2Z1 demo] Generated mini_home scene under /World/env_0/scene/mini_home', flush=True)


def clear_missing_robot_texture_references(robot_root_path='/World/env_0/robots/b2z1'):
    """Suppress known missing B2Z1 texture references in the live stage only."""
    try:
        import omni.usd
        from pxr import Usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        robot_prim = stage.GetPrimAtPath(robot_root_path)
        if not robot_prim.IsValid():
            return

        cleared = 0
        missing_fragment = 'chair2/textures/material_0.png'
        for prim in Usd.PrimRange(robot_prim):
            for attr in prim.GetAttributes():
                value = attr.Get()
                if value is None:
                    continue
                if missing_fragment in str(value):
                    attr.Block()
                    cleared += 1
        if cleared:
            print(f'[B2Z1 demo] Robot material: blocked {cleared} missing texture reference(s).', flush=True)
    except Exception as exc:
        print(f'[B2Z1 demo] Robot material cleanup failed: {exc}', flush=True)


def make_scene_background_static(scene_root_path='/World/env_0/scene'):
    """Runtime-only: keep GRScenes background as static colliders during robot physics."""
    try:
        import omni.usd
        from pxr import Usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        scene_prim = stage.GetPrimAtPath(scene_root_path)
        if not scene_prim.IsValid():
            return

        disabled = 0
        for prim in Usd.PrimRange(scene_prim):
            attr = prim.GetAttribute('physics:rigidBodyEnabled')
            if attr and attr.Get():
                attr.Set(False)
                disabled += 1
        print(f'[B2Z1 demo] Static background: disabled {disabled} scene rigid bodies at runtime.', flush=True)
    except Exception as exc:
        print(f'[B2Z1 demo] Static background failed: {exc}', flush=True)


def render_only_loop(env, args, obs, target_yaw, start_frame=0):
    """Keep the GUI responsive without advancing physics or robot controllers."""
    frame = start_frame
    while env.simulation_app.is_running():
        frame += 1
        env.warm_up(steps=1, render=True, physics=False)
        obs = env.get_observations()
        if frame % 200 == 0:
            pos = obs['position']
            roll, pitch, yaw_abs = quat_wxyz_to_rpy(obs['orientation'])
            yaw = wrap_to_pi(yaw_abs - target_yaw)
            print(
                f'  idle frame {frame:5d}: pos={np.round(pos, 3)} '
                f'rpy=({roll:.3f},{pitch:.3f},{yaw:.3f})'
            )
        if args.steps > 0 and frame >= args.steps:
            break


def describe_loaded_stage():
    """Print a compact USD-stage sanity check after reset."""
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print('[B2Z1 demo] Stage check: no active USD stage.', flush=True)
            return

        root_children = [str(prim.GetPath()) for prim in stage.GetPseudoRoot().GetChildren()]
        prim_count = sum(1 for _ in stage.Traverse())
        key_paths = [
            '/World',
            '/World/env_0',
            '/World/env_0/scene',
            '/World/env_0/robots/b2z1',
            '/World/env_0/robots/b2z1/base',
        ]
        path_status = ', '.join(
            f'{path}={"OK" if stage.GetPrimAtPath(path).IsValid() else "MISSING"}'
            for path in key_paths
        )
        print(f'[B2Z1 demo] Stage check: roots={root_children} prims={prim_count}', flush=True)
        print(f'[B2Z1 demo] Stage paths: {path_status}', flush=True)

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

        def fmt_bbox(path):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                return f'{path}=MISSING'
            box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            mn = box.GetMin()
            mx = box.GetMax()
            center = ((mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5)
            return (
                f'{path}: '
                f'min=({mn[0]:.2f},{mn[1]:.2f},{mn[2]:.2f}) '
                f'max=({mx[0]:.2f},{mx[1]:.2f},{mx[2]:.2f}) '
                f'center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})'
            )

        print(f'[B2Z1 demo] BBox {fmt_bbox("/World/env_0/scene")}', flush=True)
        print(f'[B2Z1 demo] BBox {fmt_bbox("/World/env_0/robots/b2z1/base")}', flush=True)

        ground_candidates = []
        scene_prim = stage.GetPrimAtPath('/World/env_0/scene')
        if scene_prim.IsValid():
            for prim in Usd.PrimRange(scene_prim):
                path = str(prim.GetPath())
                lower_path = path.lower()
                if '/ground/' not in lower_path:
                    continue
                try:
                    box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
                except Exception:
                    continue
                mn = box.GetMin()
                mx = box.GetMax()
                area = max(0.0, mx[0] - mn[0]) * max(0.0, mx[1] - mn[1])
                if area < 0.5:
                    continue
                center = ((mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, mx[2] + 0.6)
                ground_candidates.append((area, path, center, mn, mx))
        for area, path, center, mn, mx in sorted(ground_candidates, reverse=True)[:5]:
            print(
                '[B2Z1 demo] Ground candidate: '
                f'area={area:.2f} spawn=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) '
                f'min=({mn[0]:.2f},{mn[1]:.2f},{mn[2]:.2f}) '
                f'max=({mx[0]:.2f},{mx[1]:.2f},{mx[2]:.2f}) '
                f'path={path}',
                flush=True,
            )
    except Exception as exc:
        print(f'[B2Z1 demo] Stage check failed: {exc}', flush=True)


def resolve_scene_asset_path(args):
    if args.scene_asset_path:
        scene_path = os.path.abspath(os.path.expanduser(args.scene_asset_path))
    else:
        scene_path = os.path.join(gm.ASSET_PATH, SCENE_PRESETS[args.scene])
    if not os.path.exists(scene_path):
        raise FileNotFoundError(
            f'Scene USD does not exist: {scene_path}\n'
            f'Use --scene-asset-path to point at another local USD.'
        )
    return scene_path


def parse_args():
    p = argparse.ArgumentParser(description='B2Z1 locomotion demo with ALORE policy.')
    p.add_argument('--steps', type=int, default=0,
                   help='Steps to run (0 = run until window closed).')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--scene', choices=sorted(SCENE_PRESETS), default='empty',
                   help='Scene preset to load from InternUtopia assets_full.')
    p.add_argument('--scene-asset-path', type=str, default=None,
                   help='Override preset with an explicit local USD path.')
    p.add_argument('--scene-position', type=str, default='0,0,0',
                   help='Scene translation as x,y,z.')
    p.add_argument('--scene-scale', type=str, default='1,1,1',
                   help='Scene scale as x,y,z.')
    p.add_argument('--robot-position', type=str, default='0,0,0.6',
                   help='Robot spawn position as x,y,z.')
    p.add_argument('--forward-speed', type=float, default=0.02)
    p.add_argument('--lateral-speed', type=float, default=0.0)
    p.add_argument('--rotation-speed', type=float, default=0.0)
    p.add_argument('--stabilize-seconds', type=float, default=2.0)
    p.add_argument('--ramp-seconds', type=float, default=2.0)
    p.add_argument('--idle', action='store_true',
                   help='Render the loaded scene and robot without stepping physics or running locomotion.')
    p.add_argument('--move-seconds', type=float, default=0.0,
                   help='Run locomotion for this many seconds after stabilization, then switch to render-only idle.')
    p.add_argument('--scene-render-warmup-seconds', type=float, default=0.0,
                   help='Render-only warmup before locomotion/idle loop, so scene assets can finish loading.')
    p.add_argument('--clear-missing-robot-textures', action=argparse.BooleanOptionalAction, default=True,
                   help='Block known missing B2Z1 texture references in the live stage.')
    p.add_argument('--static-scene-background', action='store_true',
                   help='Runtime-only: make imported GRScenes rigid bodies static while preserving collisions.')
    p.add_argument('--stabilize-mode', choices=('hold', 'policy'), default='hold',
                   help='Use direct joint hold or zero-speed locomotion policy during stabilization.')
    p.add_argument('--heading-hold', action=argparse.BooleanOptionalAction, default=True,
                   help='Apply a small yaw-rate correction while walking straight.')
    p.add_argument('--heading-kp', type=float, default=1.0,
                   help='Yaw hold proportional gain [rad/s per rad].')
    p.add_argument('--max-heading-correction', type=float, default=0.3,
                   help='Clamp heading-hold yaw-rate correction [rad/s].')
    p.add_argument('--path-hold', action=argparse.BooleanOptionalAction, default=False,
                   help='Bias the held heading back toward the start line.')
    p.add_argument('--path-kp', type=float, default=1.0,
                   help='Path hold gain [rad yaw offset per m lateral error].')
    p.add_argument('--max-path-yaw', type=float, default=0.8,
                   help='Clamp path-hold yaw offset [rad].')
    p.add_argument('--lateral-hold', action=argparse.BooleanOptionalAction, default=False,
                   help='Apply a small lateral-speed correction to keep the start line.')
    p.add_argument('--lateral-kp', type=float, default=0.05,
                   help='Lateral hold proportional gain [m/s per m].')
    p.add_argument('--max-lateral-correction', type=float, default=0.08,
                   help='Clamp lateral-hold speed correction [m/s].')
    p.add_argument('--auto-camera', action=argparse.BooleanOptionalAction, default=True,
                   help='Create a debug camera and switch the GUI viewport to it.')
    p.add_argument('--force-camera-view', action='store_true',
                   help='Deprecated alias for --auto-camera.')
    return p.parse_args()


def main():
    args = parse_args()
    headless = args.headless or not has_display()
    scene_asset_path = resolve_scene_asset_path(args)
    scene_position = parse_vec3(args.scene_position, '--scene-position')
    scene_scale = parse_vec3(args.scene_scale, '--scene-scale')
    robot_position = parse_vec3(args.robot_position, '--robot-position')

    config = Config(
        simulator=SimConfig(
            physics_dt=1 / 200,   # match ALORE sim.dt = 0.005
            rendering_dt=1 / 50,
            use_fabric=False,
            headless=headless,
            webrtc=headless,
        ),
        task_configs=[
            SingleInferenceTaskCfg(
                scene_asset_path=scene_asset_path,
                scene_position=scene_position,
                scene_scale=scene_scale,
                robots=[
                    B2Z1RobotCfg(
                        position=robot_position,  # match ALORE B2Z1 spawn height
                        controllers=[move_by_speed_cfg, arm_controller_cfg, body_hold_controller_cfg],
                    )
                ],
            )
        ],
    )

    import_extensions()
    env = Env(config)
    obs, _ = env.reset()
    if args.clear_missing_robot_textures:
        clear_missing_robot_texture_references()
    if args.static_scene_background:
        make_scene_background_static()
    if args.scene in GENERATED_SCENES:
        add_mini_home_scene()
    describe_loaded_stage()

    print('=== B2Z1 Locomotion Demo ===', flush=True)
    print(f'  Scene       : {args.scene} ({scene_asset_path})', flush=True)
    print(f'  Scene pos   : {scene_position}  scale={scene_scale}', flush=True)
    print(f'  Robot start : {np.round(obs["position"], 3)}', flush=True)
    print(f'  Speed cmd   : fwd={args.forward_speed}  lat={args.lateral_speed}  rot={args.rotation_speed}', flush=True)
    print(f'  Heading hold: {args.heading_hold}  kp={args.heading_kp}  max={args.max_heading_correction}', flush=True)
    print(f'  Path hold   : {args.path_hold}  kp={args.path_kp}  max_yaw={args.max_path_yaw}', flush=True)
    print(f'  Lateral hold: {args.lateral_hold}  kp={args.lateral_kp}  max={args.max_lateral_correction}', flush=True)
    if not headless:
        apply_gui_view(obs['position'], auto_camera=(args.auto_camera or args.force_camera_view))

    if args.scene_render_warmup_seconds > 0:
        warmup_steps = max(1, int(args.scene_render_warmup_seconds * 50))
        print(
            f'[B2Z1 demo] Scene render warmup: {args.scene_render_warmup_seconds:.1f}s '
            f'({warmup_steps} render-only frames, no physics/action).',
            flush=True,
        )
        env.warm_up(steps=warmup_steps, render=True, physics=False)
        obs = env.get_observations()

    step = 0
    sim_hz = int(round(1 / env.get_dt()))
    stabilize_steps = max(0, int(args.stabilize_seconds * sim_hz))
    ramp_steps = max(1, int(args.ramp_seconds * sim_hz))
    target_speed_cmd = np.array([args.forward_speed, args.lateral_speed, args.rotation_speed], dtype=np.float32)
    translation_commanded = np.linalg.norm(target_speed_cmd[:2]) > 1e-6
    start_position = np.array(obs['position'], dtype=float)
    target_yaw = quat_wxyz_to_yaw(obs['orientation'])
    target_left_axis = np.array([-np.sin(target_yaw), np.cos(target_yaw)], dtype=float)

    try:
        if args.idle:
            print('[B2Z1 demo] Idle mode: rendering only; locomotion policy is not stepped.', flush=True)
            render_only_loop(env, args, obs, target_yaw)
            return

        move_stop_step = 0
        if args.move_seconds > 0:
            move_stop_step = stabilize_steps + max(1, int(args.move_seconds * sim_hz))
            print(
                f'[B2Z1 demo] Safe move window: {args.move_seconds:.2f}s after '
                f'{args.stabilize_seconds:.2f}s stabilization, then render-only idle.',
                flush=True,
            )

        while env.simulation_app.is_running():
            step += 1
            current_yaw = quat_wxyz_to_yaw(obs['orientation'])
            delta_xy = np.array(obs['position'][:2], dtype=float) - start_position[:2]
            lateral_error = float(np.dot(delta_xy, target_left_axis))
            target_yaw_cmd = target_yaw
            if step <= stabilize_steps:
                speed_cmd = [0.0, 0.0, 0.0]
                if args.stabilize_mode == 'hold':
                    action = {
                        'body_hold': [BODY_HOLD],
                    }
                    obs, _, terminated, _, _ = env.step(action=action)
                    if step % 200 == 0:
                        pos = obs['position']
                        roll, pitch, yaw_abs = quat_wxyz_to_rpy(obs['orientation'])
                        yaw = wrap_to_pi(yaw_abs - target_yaw)
                        print(
                            f'  step {step:5d}: pos={np.round(pos, 3)} '
                            f'rpy=({roll:.3f},{pitch:.3f},{yaw:.3f}) mode=hold'
                        )
                    if (args.steps > 0 and step >= args.steps) or terminated:
                        break
                    continue
            else:
                ramp_alpha = min(1.0, (step - stabilize_steps) / ramp_steps)
                speed_cmd = (target_speed_cmd * ramp_alpha).tolist()
                if args.heading_hold and translation_commanded and abs(args.rotation_speed) < 1e-6:
                    if args.path_hold and abs(args.lateral_speed) < 1e-6:
                        yaw_offset = np.clip(
                            -args.path_kp * lateral_error,
                            -args.max_path_yaw,
                            args.max_path_yaw,
                        )
                        target_yaw_cmd = wrap_to_pi(target_yaw + float(yaw_offset))
                    yaw_error = wrap_to_pi(current_yaw - target_yaw_cmd)
                    yaw_correction = np.clip(
                        -args.heading_kp * yaw_error,
                        -args.max_heading_correction,
                        args.max_heading_correction,
                    )
                    speed_cmd[2] += float(yaw_correction)
                if args.lateral_hold and abs(args.lateral_speed) < 1e-6:
                    lateral_correction = np.clip(
                        -args.lateral_kp * lateral_error,
                        -args.max_lateral_correction,
                        args.max_lateral_correction,
                    )
                    speed_cmd[1] += float(lateral_correction)
            action = {
                'move_by_speed': speed_cmd,
                'arm_control': [ARM_HOME],   # hold arm in home position
            }
            obs, _, terminated, _, _ = env.step(action=action)

            if step % 200 == 0:
                pos = obs['position']
                roll, pitch, yaw_abs = quat_wxyz_to_rpy(obs['orientation'])
                yaw = wrap_to_pi(yaw_abs - target_yaw)
                lateral_error = float(np.dot(np.array(pos[:2], dtype=float) - start_position[:2], target_left_axis))
                yaw_target = wrap_to_pi(target_yaw_cmd - target_yaw)
                print(
                    f'  step {step:5d}: pos={np.round(pos, 3)} '
                    f'rpy=({roll:.3f},{pitch:.3f},{yaw:.3f}) '
                    f'yaw_tgt={yaw_target:.3f} yerr={lateral_error:.3f} '
                    f'lat_cmd={speed_cmd[1]:.3f} rot_cmd={speed_cmd[2]:.3f}'
                )

            if move_stop_step > 0 and step >= move_stop_step:
                print('[B2Z1 demo] Move window finished; switching to render-only idle.', flush=True)
                render_only_loop(env, args, obs, target_yaw, start_frame=step)
                break

            if (args.steps > 0 and step >= args.steps) or terminated:
                break
    finally:
        print('Demo finished.')
        env.close()


if __name__ == '__main__':
    main()

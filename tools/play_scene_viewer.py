"""Load a GRScenes USD in Isaac Sim with a usable inspection camera.

This is a pure scene viewer: it does not add robots or save edits back to USD.
It only creates an in-memory camera and optionally hides ceiling/HDR prims so
closed indoor scenes are easier to inspect from the GUI.
"""

import argparse
import os


def parse_vec3(text, arg_name):
    try:
        values = tuple(float(item.strip()) for item in text.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'{arg_name} must be three comma-separated numbers.') from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f'{arg_name} must have exactly three values.')
    return values


def parse_args():
    parser = argparse.ArgumentParser(description='Load a GRScenes USD with a debug inspection camera.')
    parser.add_argument('-f', '--file', required=True, type=str, help='USD file to open')
    parser.add_argument('--camera-target', type=str, default=None, help='Override target as x,y,z in stage units')
    parser.add_argument('--camera-eye', type=str, default=None, help='Override camera eye as x,y,z in stage units')
    parser.add_argument('--auto-target', action='store_true',
                        help='Scan the scene for a ground target. This can be memory-heavy on large GRScenes.')
    parser.add_argument('--hide-ceiling', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--hide-hdr', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def make_invisible(prim):
    from pxr import UsdGeom

    if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
        UsdGeom.Imageable(prim).MakeInvisible()
        return True
    return False


def apply_visibility_filters(stage, hide_ceiling=True, hide_hdr=True):
    hidden = []
    if hide_hdr:
        for path in ('/Root/default_setting/HDR_Sphere', '/Root/default_setting/HDR_Sphere/SM_Sphere'):
            prim = stage.GetPrimAtPath(path)
            if make_invisible(prim):
                hidden.append(str(prim.GetPath()))

    if hide_ceiling:
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            lower_path = path.lower()
            if '/ceiling/' in lower_path or lower_path.endswith('/ceiling'):
                if make_invisible(prim):
                    hidden.append(path)

    print(f'[GRScenes viewer] Hidden prims: {len(hidden)}', flush=True)
    for path in hidden[:12]:
        print(f'[GRScenes viewer]   hidden {path}', flush=True)
    if len(hidden) > 12:
        print(f'[GRScenes viewer]   ... {len(hidden) - 12} more', flush=True)


def fallback_camera_for_scene(scene_path):
    """Known-good inspection camera presets in the original GRScenes centimeter units."""
    normalized = scene_path.replace('\\', '/')
    if 'MWBGLKQKTKJZ2AABAAAAACA8_usd' in normalized:
        # Same floor candidate used by the B2Z1 demo, converted from meters to
        # the source scene's centimeter units.
        return (-142.0, -271.0, 90.0), (158.0, -671.0, 282.0), 0.01
    return None


def find_scene_target(stage):
    from pxr import Usd, UsdGeom

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if not meters_per_unit or meters_per_unit <= 0.0:
        meters_per_unit = 0.01
    human_eye_units = 1.35 / meters_per_unit

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    ground_candidates = []
    furniture_candidates = []
    root_prim = stage.GetDefaultPrim()
    if not root_prim or not root_prim.IsValid():
        root_prim = stage.GetPrimAtPath('/Root')
    search_root = root_prim if root_prim and root_prim.IsValid() else stage.GetPseudoRoot()

    for prim in Usd.PrimRange(search_root):
        path = str(prim.GetPath())
        lower_path = path.lower()
        try:
            box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        except Exception:
            continue
        mn = box.GetMin()
        mx = box.GetMax()
        dx = max(0.0, mx[0] - mn[0])
        dy = max(0.0, mx[1] - mn[1])
        dz = max(0.0, mx[2] - mn[2])
        area = dx * dy
        if area <= 0.0:
            continue

        center = (
            (mn[0] + mx[0]) * 0.5,
            (mn[1] + mx[1]) * 0.5,
            mx[2] + human_eye_units,
        )
        if '/ground/' in lower_path and area > (0.2 / meters_per_unit) ** 2:
            ground_candidates.append((area, path, center, mn, mx))
        if '/furnitures/' in lower_path and dz > 0.05 / meters_per_unit:
            furniture_candidates.append((area, path, center, mn, mx))

    candidates = ground_candidates or furniture_candidates
    if candidates:
        area, path, center, mn, mx = sorted(candidates, reverse=True)[0]
        print(
            '[GRScenes viewer] Camera target candidate: '
            f'area={area:.2f} target=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) '
            f'min=({mn[0]:.2f},{mn[1]:.2f},{mn[2]:.2f}) '
            f'max=({mx[0]:.2f},{mx[1]:.2f},{mx[2]:.2f}) '
            f'path={path}',
            flush=True,
        )
        return center, meters_per_unit

    box = bbox_cache.ComputeWorldBound(search_root).ComputeAlignedBox()
    mn = box.GetMin()
    mx = box.GetMax()
    center = (
        (mn[0] + mx[0]) * 0.5,
        (mn[1] + mx[1]) * 0.5,
        (mn[2] + mx[2]) * 0.5,
    )
    print(f'[GRScenes viewer] Fallback camera target: {center}', flush=True)
    return center, meters_per_unit


def apply_debug_camera(stage, target, eye):
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Gf, Sdf, UsdGeom

    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except ModuleNotFoundError:
        from omni.isaac.core.utils.viewports import set_camera_view

    camera_path = '/Root/GRScenesDebugCamera'
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000000.0))
    camera.CreateFocalLengthAttr(16.0)

    viewport = get_active_viewport()
    if viewport is None:
        print('[GRScenes viewer] No active viewport; camera was created but not selected.', flush=True)
        return

    set_camera_view(eye=eye, target=target, camera_prim_path=camera_path, viewport_api=viewport)
    try:
        viewport.camera_path = camera_path
    except TypeError:
        viewport.camera_path = Sdf.Path(camera_path)
    print(
        '[GRScenes viewer] GUI camera: '
        f'{camera_path} eye={[round(v, 3) for v in eye]} target={[round(v, 3) for v in target]}',
        flush=True,
    )


def main():
    args = parse_args()
    f_abs_path = os.path.abspath(args.file)
    if not os.path.exists(f_abs_path):
        raise FileNotFoundError(f'USD file not found: {args.file}')

    from isaacsim import SimulationApp

    kit = SimulationApp(launch_config={'headless': False})

    import omni.physx.bindings._physx as physx_bindings

    kit.set_setting(physx_bindings.SETTING_UJITSO_COLLISION_COOKING, False)
    kit.context.open_stage(f_abs_path)
    stage = kit.context.get_stage()
    kit._wait_for_viewport()

    apply_visibility_filters(stage, hide_ceiling=args.hide_ceiling, hide_hdr=args.hide_hdr)

    preset = fallback_camera_for_scene(f_abs_path)
    if args.camera_target:
        target = parse_vec3(args.camera_target, '--camera-target')
        meters_per_unit = 0.01
    elif preset is not None:
        target, _, meters_per_unit = preset
        print(f'[GRScenes viewer] Camera preset target: {target}', flush=True)
    elif args.auto_target:
        target, meters_per_unit = find_scene_target(stage)
    else:
        target = (0.0, 0.0, 100.0)
        meters_per_unit = 0.01
        print(f'[GRScenes viewer] Default camera target: {target}', flush=True)

    if args.camera_eye:
        eye = parse_vec3(args.camera_eye, '--camera-eye')
    elif preset is not None and not args.camera_target:
        _, eye, _ = preset
        print(f'[GRScenes viewer] Camera preset eye: {eye}', flush=True)
    else:
        eye = (
            target[0] + 2.4 / meters_per_unit,
            target[1] - 3.0 / meters_per_unit,
            target[2] + 1.1 / meters_per_unit,
        )
    apply_debug_camera(stage, target, eye)

    import omni.timeline

    omni.timeline.get_timeline_interface().play()
    while kit.is_running():
        kit.update()


if __name__ == '__main__':
    main()

"""B2Z1 robot class for InternUtopia — adapted from AliengoZ1Robot."""

from collections import OrderedDict
from pathlib import Path

import numpy as np

from internutopia.core.robot.articulation import IArticulation
from internutopia.core.robot.rigid_body import IRigidBody
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia.core.util import log
from internutopia_extension.configs.robots.b2z1 import B2Z1RobotCfg

# B2 standing pose in Isaac Sim joint order (19 joints, no gripper adjustment)
_B2_STANDING_JOINTS = [
    'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
    'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
    'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint',
]
_B2_STANDING_POS = np.array([
    0.1, -0.1, 0.1, -0.1,    # hip
    0.8, 0.8, 0.8, 0.8,       # thigh
    -1.5, -1.5, -1.5, -1.5,   # calf
], dtype=np.float32)

_Z1_ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'jointGripper']
_Z1_ARM_HOME = np.array([0.0, 1.48, -0.63, -0.84, 0.0, 1.57, -1.2], dtype=np.float32)


@BaseRobot.register('B2Z1Robot')
class B2Z1Robot(BaseRobot):
    def __init__(self, config: B2Z1RobotCfg, scene: IScene):
        super().__init__(config, scene)
        self._start_position = np.array(config.position) if config.position is not None else None
        self._start_orientation = np.array(config.orientation) if config.orientation is not None else None
        self._robot_base: IRigidBody | None = None
        self._robot_scale = np.array(config.scale) if config.scale is not None else np.array([1.0, 1.0, 1.0])

        log.debug(f'b2z1 {config.name}: position    : {self._start_position}')
        log.debug(f'b2z1 {config.name}: usd_path    : {config.usd_path}')
        log.debug(f'b2z1 {config.name}: prim_path   : {config.prim_path}')

        articulation_prim_path = self._load_usd_and_find_articulation_root(config)
        self.articulation = IArticulation.create(
            prim_path=articulation_prim_path,
            name=config.name,
            position=None,
            orientation=None,
            usd_path=None,
            scale=None,
        )

    def _load_usd_and_find_articulation_root(self, config: B2Z1RobotCfg) -> str:
        from isaacsim.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from pxr import Gf, Usd, UsdGeom

        usd_path = str(Path(config.usd_path).resolve())
        if not Path(usd_path).is_file():
            raise FileNotFoundError(f'B2Z1 USD not found: {usd_path}')

        root_prim = get_prim_at_path(config.prim_path)
        if not root_prim.IsValid():
            add_reference_to_stage(prim_path=config.prim_path, usd_path=usd_path)
            root_prim = get_prim_at_path(config.prim_path)

        xformable = UsdGeom.Xformable(root_prim)
        xformable.ClearXformOpOrder()
        if self._start_position is not None:
            p = self._start_position
            xformable.AddTranslateOp().Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
        if self._start_orientation is not None:
            o = self._start_orientation
            xformable.AddOrientOp().Set(Gf.Quatd(float(o[0]), float(o[1]), float(o[2]), float(o[3])))
        if not np.allclose(self._robot_scale, 1.0):
            s = self._robot_scale
            xformable.AddScaleOp().Set(Gf.Vec3f(float(s[0]), float(s[1]), float(s[2])))

        articulation_roots = [
            str(prim.GetPath())
            for prim in Usd.PrimRange.AllPrims(root_prim)
            if 'PhysicsArticulationRootAPI' in prim.GetAppliedSchemas()
        ]
        if articulation_roots:
            articulation_root = min(articulation_roots, key=len)
        else:
            articulation_root = config.prim_path
            log.warning(f'No PhysicsArticulationRootAPI found under {config.prim_path}; using root.')
            return articulation_root

        stage = root_prim.GetStage()
        articulation_root = self._convert_to_floating_base(stage, root_prim, articulation_root, config.prim_path)
        return articulation_root

    @staticmethod
    def _convert_to_floating_base(stage, root_prim, articulation_root_path: str, robot_root_path: str) -> str:
        from pxr import Usd, UsdPhysics

        ar_prim = stage.GetPrimAtPath(articulation_root_path)
        body0_rel = ar_prim.GetRelationship('physics:body0')
        if not body0_rel or body0_rel.GetTargets():
            log.info(f'b2z1: articulation root {articulation_root_path} already floating-base')
            return articulation_root_path

        base_path = None
        body1_rel = ar_prim.GetRelationship('physics:body1')
        if body1_rel:
            targets = body1_rel.GetTargets()
            if targets:
                base_path = str(targets[0])

        if not base_path:
            for prim in Usd.PrimRange.AllPrims(root_prim):
                if 'PhysicsRigidBodyAPI' in prim.GetAppliedSchemas():
                    base_path = str(prim.GetPath())
                    break

        if not base_path:
            log.warning('b2z1: could not find base rigid body; keeping original root')
            return articulation_root_path

        base_prim = stage.GetPrimAtPath(base_path)
        if not base_prim.IsValid():
            log.warning(f'b2z1: base prim {base_path} not valid')
            return articulation_root_path

        UsdPhysics.ArticulationRootAPI.Apply(base_prim)
        log.info(f'b2z1: applied PhysicsArticulationRootAPI to base: {base_path}')

        try:
            from pxr import PhysxSchema
            old_physx = PhysxSchema.PhysxArticulationAPI(ar_prim)
            PhysxSchema.PhysxArticulationAPI.Apply(base_prim)
            if old_physx:
                for attr in ar_prim.GetAttributes():
                    if attr.GetNamespace() == 'physxArticulation' and attr.IsAuthored():
                        base_prim.CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
        except Exception as e:
            log.warning(f'b2z1: could not apply PhysxArticulationAPI: {e}')

        ar_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        joint_enabled = ar_prim.GetAttribute('physics:jointEnabled')
        if joint_enabled:
            joint_enabled.Set(False)

        log.info(f'b2z1: converted to floating-base, root at {base_path}')
        return base_path

    def create_rigid_bodies(self):
        from pxr import Usd

        stage = self._scene.unwrap().stage
        _prim = stage.GetPrimAtPath(self.config.prim_path)
        for prim in Usd.PrimRange.AllPrims(_prim):
            schemas = prim.GetAppliedSchemas()
            has_rb = 'PhysicsRigidBodyAPI' in schemas or bool(prim.GetAttribute('physics:rigidBodyEnabled'))
            if has_rb:
                _rb = IRigidBody.create(prim_path=str(prim.GetPath()), name=str(prim.GetPath()))
                self._rigid_body_map[str(prim.GetPath())] = _rb

    def post_reset(self):
        super().post_reset()
        self._robot_base = self._resolve_robot_base()
        self.articulation.set_solver_position_iteration_count(8)
        self.articulation.set_solver_velocity_iteration_count(2)
        self.articulation.set_enabled_self_collisions(False)
        self._set_gains()
        self._set_standing_pose()

    def _set_standing_pose(self):
        from internutopia.core.robot.articulation_subset import ArticulationSubset

        try:
            leg_subset = ArticulationSubset(self.articulation, _B2_STANDING_JOINTS)
            self.articulation.set_joint_positions(
                positions=_B2_STANDING_POS,
                joint_indices=leg_subset.joint_indices,
            )
            arm_subset = ArticulationSubset(self.articulation, _Z1_ARM_JOINTS)
            self.articulation.set_joint_positions(
                positions=_Z1_ARM_HOME,
                joint_indices=arm_subset.joint_indices,
            )
            log.debug('b2z1: standing + arm home pose applied')
        except Exception as e:
            log.warning(f'b2z1: could not set initial pose: {e}')

    def _set_gains(self):
        from internutopia.core.robot.articulation_subset import ArticulationSubset

        try:
            leg_subset = ArticulationSubset(self.articulation, _B2_STANDING_JOINTS)
            leg_kps = np.array([700.0] * 12)
            leg_kds = np.array([10.0] * 12)
            self.articulation.set_gains(kps=leg_kps, kds=leg_kds, joint_indices=leg_subset.joint_indices)
        except Exception as e:
            log.warning(f'b2z1: could not set leg gains: {e}')

        try:
            arm_joints_no_grip = _Z1_ARM_JOINTS[:6]
            arm_subset = ArticulationSubset(self.articulation, arm_joints_no_grip)
            arm_kps = np.array([512.0, 768.0, 768.0, 512.0, 384.0, 256.0])
            arm_kds = np.array([25.6] * 6)
            self.articulation.set_gains(kps=arm_kps, kds=arm_kds, joint_indices=arm_subset.joint_indices)
        except Exception as e:
            log.warning(f'b2z1: could not set arm gains: {e}')

        try:
            gripper_subset = ArticulationSubset(self.articulation, ['jointGripper'])
            self.articulation.set_gains(
                kps=np.array([512.0]),
                kds=np.array([25.6]),
                joint_indices=gripper_subset.joint_indices,
            )
        except Exception as e:
            log.warning(f'b2z1: could not set gripper gains: {e}')

    def _resolve_robot_base(self) -> IRigidBody:
        prim_path = self.config.prim_path.rstrip('/')
        for suffix in self.config.base_link_suffixes:
            candidate = prim_path + suffix
            if candidate in self._rigid_body_map:
                log.info(f'b2z1 base link resolved: {candidate}')
                return self._rigid_body_map[candidate]

        body_paths = sorted(self._rigid_body_map.keys())
        for bp in body_paths:
            last = bp.rsplit('/', 1)[-1].lower()
            if last in ('base', 'trunk'):
                log.info(f'b2z1 base link fallback: {bp}')
                return self._rigid_body_map[bp]

        if not body_paths:
            raise RuntimeError(f'No rigid bodies under {self.config.prim_path}')

        log.warning(f'b2z1: using first rigid body as base: {body_paths[0]}')
        return self._rigid_body_map[body_paths[0]]

    def get_robot_scale(self):
        return self._robot_scale

    def get_robot_base(self) -> IRigidBody:
        if self._robot_base is None:
            raise RuntimeError('B2Z1Robot base has not been initialized (call post_reset first).')
        return self._robot_base

    def get_pose(self):
        return self.get_robot_base().get_pose()

    def apply_action(self, action: dict):
        if not action:
            return
        for ctrl_name, ctrl_action in action.items():
            if ctrl_name not in self.controllers:
                log.warning(f'b2z1: unknown controller "{ctrl_name}"')
                continue
            ctrl = self.controllers[ctrl_name]
            control = ctrl.action_to_control(ctrl_action)
            self.articulation.apply_action(control)

    def get_obs(self) -> OrderedDict:
        position, orientation = self.get_robot_base().get_pose()
        obs = {
            'position': position,
            'orientation': orientation,
            'controllers': {},
            'sensors': {},
        }
        for c_name, ctrl in self.controllers.items():
            obs['controllers'][c_name] = ctrl.get_obs()
        for s_name, sensor in self.sensors.items():
            obs['sensors'][s_name] = sensor.get_data()
        return self._make_ordered(obs)

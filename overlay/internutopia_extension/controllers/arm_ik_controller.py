"""ArmIKController: drives B2Z1's Z1 arm to a world-frame EE target via Pinocchio IK."""

from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from internutopia.core.robot.articulation_action import ArticulationAction
from internutopia.core.robot.articulation_subset import ArticulationSubset
from internutopia.core.robot.controller import BaseController
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia.core.util import log
from internutopia_extension.configs.controllers.arm_ik_controller import ArmIKControllerCfg
from internutopia_extension.controllers.z1_ik_solver import Z1IKSolver

# B2Z1 arm joint default pose (matches ALORE joint_default for arm)
_ARM_DEFAULT = np.array([0.0, 1.48, -0.63, -0.84, 0.0, 1.57, 0.0])  # joint1-6 + gripper


@BaseController.register('ArmIKController')
class ArmIKController(BaseController):
    """Drives the Z1 arm to a world-frame EE target [x,y,z,qx,qy,qz,qw,gripper_angle].

    Action format:
        [x, y, z, qx, qy, qz, qw, gripper_angle]

        - (x,y,z)          : EE target position in WORLD frame [m]
        - (qx,qy,qz,qw)    : EE target orientation quaternion in WORLD frame
        - gripper_angle     : gripper joint angle [rad], -1.57=open, 0=closed

    If only 7 elements, gripper stays unchanged.
    If only 3 elements, orientation is inferred from current arm pose.
    """

    def __init__(self, config: ArmIKControllerCfg, robot: BaseRobot, scene: IScene) -> None:
        super().__init__(config=config, robot=robot, scene=scene)

        self._ik = Z1IKSolver(urdf_path=config.urdf_path) if config.urdf_path else Z1IKSolver()
        self.joint_subset: ArticulationSubset | None = None
        if config.joint_names:
            self.joint_subset = ArticulationSubset(self.robot.articulation, config.joint_names)

        self._arm_base_prim_path = self.robot.config.prim_path.rstrip('/') + config.arm_base_prim_suffix
        self._arm_base_prim = None  # lazy-initialised after scene loads

        self._current_q = _ARM_DEFAULT.copy()
        self._gripper_angle = -1.57    # open

        # ── 自适应夹持状态 ──────────────────────────────────────────────
        # 夹爪从 -1.5(张开) 往 -0.05(闭合) 给目标；若实际角度追不上目标
        # (被物体挡住、卡住不动)，判定夹住物体表面，停止继续闭合并保持。
        # 这样物体多大都能夹住，不会盲目合到固定角度 → 解决"太大抓不住"。
        self._gripper_jammed = False          # 是否已夹住物体
        self._gripper_hold_angle = None       # 夹住时锁定的保持目标
        self._prev_gripper_actual = None      # 上一帧夹爪实际角度
        self._jam_pos_eps = 0.02              # 帧间实际位置变化小于此值 → 不动
        self._jam_err_eps = 0.08              # 命令与实际偏差大于此值 → 卡住
        self._gripper_hold_margin = 0.15      # 保持夹持力的余量(往闭合方向再压一点)

    def _get_arm_base_world_pose(self):
        """Return (pos_world (3,), rot_world (3,3)) for the arm base link (link00)."""
        if self._arm_base_prim is None:
            from omni.isaac.core.prims import XFormPrim
            self._arm_base_prim = XFormPrim(prim_path=self._arm_base_prim_path)

        pos, quat_wxyz = self._arm_base_prim.get_world_pose()
        # Isaac Sim returns (w,x,y,z); Rotation expects (x,y,z,w)
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rot = Rotation.from_quat(quat_xyzw).as_matrix()
        return np.array(pos), rot

    def forward(self, target: np.ndarray) -> ArticulationAction:
        """Compute arm joint targets for a given world-frame EE target.

        Args:
            target: ndarray of shape (3,), (7,), or (8,)
                    [x, y, z] or [x, y, z, qx, qy, qz, qw] or [x, y, z, qx, qy, qz, qw, gripper]
        """
        target = np.asarray(target, dtype=np.float64).ravel()

        # Parse target pose
        pos_world = target[:3]
        if len(target) >= 7:
            quat_xyzw = target[3:7]  # (qx,qy,qz,qw)
            rot_world = Rotation.from_quat(quat_xyzw).as_matrix()
        else:
            # Use current EE orientation from FK
            _, rot_world = self._ik.fk(self._current_q)
            # fk gives pose in link00 frame; transform back to world
            try:
                base_pos, base_rot = self._get_arm_base_world_pose()
                rot_world = base_rot @ rot_world
            except Exception:
                rot_world = np.eye(3)

        if len(target) >= 8:
            self._gripper_angle = self._adaptive_gripper(float(target[7]))

        # Transform target from world to arm base frame
        try:
            base_pos, base_rot = self._get_arm_base_world_pose()
        except Exception as e:
            log.warning(f'arm_ik: could not get arm base pose: {e}')
            return self._hold_current()

        pos_base, rot_base = Z1IKSolver.world_to_base_frame(
            pos_world, rot_world, base_pos, base_rot
        )

        # Solve IK
        q_arm, err = self._ik.solve(pos_base, rot_base, q_init=self._current_q[:6].copy())
        if err > 0.05:
            log.warning(f'arm_ik: IK did not converge well (err={err:.4f})')

        q_full = np.append(q_arm[:6], self._gripper_angle)
        self._current_q = q_full.copy()

        if self.joint_subset is None:
            return ArticulationAction(joint_positions=q_full)
        return self.joint_subset.make_articulation_action(
            joint_positions=q_full, joint_velocities=None
        )

    def _read_gripper_actual(self) -> Optional[float]:
        """读取夹爪关节(jointGripper)的实际角度；joint_names 最后一个即夹爪。"""
        if self.joint_subset is None:
            return None
        try:
            q = self.joint_subset.get_joint_positions()
            return float(q[-1])
        except Exception:
            return None

    def _adaptive_gripper(self, cmd_angle: float) -> float:
        """自适应夹持：闭合时若夹爪卡住(被物体挡住)，停止继续合并保持夹持力。

        约定：-1.5≈张开，-0.05≈完全闭合（角度越大越合）。
        闭合方向 = cmd_angle 比当前实际角度更大（更靠近 0）。

        Returns:
            实际下发给夹爪的目标角度。
        """
        actual = self._read_gripper_actual()

        # 读不到实际位置时退化为直接透传命令
        if actual is None:
            self._prev_gripper_actual = None
            return cmd_angle

        # 命令在张开方向（cmd 比实际更小/更负）→ 重置夹持状态，正常透传
        if cmd_angle <= actual + 1e-3:
            self._gripper_jammed = False
            self._gripper_hold_angle = None
            self._prev_gripper_actual = actual
            return cmd_angle

        # 已判定夹住 → 锁定保持目标，不再继续闭合
        if self._gripper_jammed and self._gripper_hold_angle is not None:
            self._prev_gripper_actual = actual
            return self._gripper_hold_angle

        # 闭合中：检测是否卡住
        # 条件：命令想合(cmd>actual 较多) 且 实际位置帧间几乎不动
        cmd_err = cmd_angle - actual
        moved = (abs(actual - self._prev_gripper_actual)
                 if self._prev_gripper_actual is not None else 1.0)
        self._prev_gripper_actual = actual

        if cmd_err > self._jam_err_eps and moved < self._jam_pos_eps:
            # 夹住物体表面：锁定在“实际位置再往里压一点”以维持夹持力
            self._gripper_jammed = True
            self._gripper_hold_angle = float(np.clip(
                actual + self._gripper_hold_margin, -1.5, -0.05))
            log.info(f'arm_ik: 夹爪夹住物体 (actual={actual:.3f}, '
                     f'保持目标={self._gripper_hold_angle:.3f})')
            return self._gripper_hold_angle

        return cmd_angle

    def reset_gripper(self):
        """松开/重置自适应夹持状态（放下物体后调用）。"""
        self._gripper_jammed = False
        self._gripper_hold_angle = None
        self._prev_gripper_actual = None

    def _hold_current(self) -> ArticulationAction:
        if self.joint_subset is None:
            return ArticulationAction(joint_positions=self._current_q)
        return self.joint_subset.make_articulation_action(
            joint_positions=self._current_q, joint_velocities=None
        )

    def action_to_control(self, action: list | np.ndarray) -> ArticulationAction:
        return self.forward(np.asarray(action))

    def get_obs(self) -> dict:
        return {
            'q': self._current_q.tolist(),
            'gripper': self._gripper_angle,
            'gripper_jammed': self._gripper_jammed,   # True = 已夹住物体
        }

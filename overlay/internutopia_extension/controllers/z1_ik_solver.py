"""Pinocchio-based IK solver for the Unitree Z1 arm (6 DOF + gripper)."""

from pathlib import Path
from typing import Optional

import numpy as np

_Z1_URDF_PATH = str(
    Path(__file__).resolve().parents[2]
    / 'local_assets' / 'robots' / 'z1' / 'urdf' / 'z1_original.urdf'
)

_EE_FRAME   = 'gripperStator'
_BASE_FRAME = 'link00'

# IK hyper-params (same as ALORE)
_IK_DT      = 3e-3
_IK_IT_MAX  = 200
_IK_EPS     = 1e-3
_IK_DAMP    = 1e-12


class Z1IKSolver:
    """Iterative IK for the Z1 arm using Pinocchio damped-least-squares."""

    def __init__(self, urdf_path: str = _Z1_URDF_PATH):
        try:
            import pinocchio as pin
            self._pin = pin
        except ImportError:
            raise ImportError(
                'pinocchio is required for Z1IKSolver. '
                'Install it with: pip install pin'
            )

        self._model = self._pin.buildModelFromUrdf(urdf_path)
        self._data  = self._model.createData()

        self._ee_id   = self._model.getFrameId(_EE_FRAME,  self._pin.FrameType.BODY)
        self._base_id = self._model.getFrameId(_BASE_FRAME, self._pin.FrameType.BODY)

        # default arm pose (folded, matches ALORE joint_default minus gripper)
        self._default_q = np.array([0.0, 1.48, -0.63, -0.84, 0.0, 1.57, 0.0])

    def solve(
        self,
        target_pos_base: np.ndarray,
        target_rot_base: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        max_iter: int = _IK_IT_MAX,
    ) -> tuple[np.ndarray, float]:
        """Solve IK for a target EE pose expressed in the arm base (link00) frame.

        Args:
            target_pos_base: (3,) position in link00 frame [m]
            target_rot_base: (3,3) rotation matrix in link00 frame
            q_init: initial joint angles (7,), uses default if None
            max_iter: max IK iterations

        Returns:
            q: (7,) joint angles [joint1..6, jointGripper]  (gripper unchanged)
            err_norm: final error norm (lower is better, <1e-3 = converged)
        """
        pin = self._pin
        oMdes = pin.SE3(target_rot_base.copy(), target_pos_base.copy())

        q = (self._default_q.copy() if q_init is None else q_init.copy())
        err_norm = 1e9

        for _ in range(max_iter):
            pin.forwardKinematics(self._model, self._data, q)
            pin.updateFramePlacements(self._model, self._data)
            iMd = self._data.oMf[self._ee_id].actInv(oMdes)
            err = pin.log(iMd).vector
            err_norm = float(np.linalg.norm(err))
            if err_norm < _IK_EPS:
                break
            J = pin.computeFrameJacobian(
                self._model, self._data, q, self._ee_id,
                pin.ReferenceFrame.LOCAL,
            )
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + _IK_DAMP * np.eye(6), err))
            q = pin.integrate(self._model, q, v * _IK_DT)

        return q, err_norm

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: return EE position and rotation in link00 frame."""
        self._pin.forwardKinematics(self._model, self._data, q)
        self._pin.updateFramePlacements(self._model, self._data)
        ee_se3 = self._data.oMf[self._ee_id]
        return ee_se3.translation.copy(), ee_se3.rotation.copy()

    @staticmethod
    def grasp_pose_in_world(
        obj_pos_world: np.ndarray,
        obj_yaw: float,
        dx: float,
        dz: float,
        pitch_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute grasp EE target in world frame from object pose.

        This mirrors ALORE's get_grasp_pose() logic.

        Args:
            obj_pos_world: (3,) object position in world frame
            obj_yaw: object yaw angle [rad]
            dx: approach distance (front offset from object centre) [m]
            dz: height offset above object centre [m]
            pitch_deg: gripper pitch angle [deg]

        Returns:
            pos_world: (3,) target EE position in world frame
            rot_world: (3,3) target EE rotation in world frame
        """
        x_c, y_c, z_c = obj_pos_world
        pos_world = np.array([
            x_c - dx * np.cos(obj_yaw),
            y_c - dx * np.sin(obj_yaw),
            z_c + dz,
        ])

        pitch = np.deg2rad(pitch_deg)
        cy, sy = np.cos(obj_yaw), np.sin(obj_yaw)
        cp, sp = np.cos(pitch),   np.sin(pitch)
        # rotation: first pitch around local Y, then yaw around Z
        rot_world = np.array([
            [ cy*cp,  -sy,  cy*sp],
            [ sy*cp,   cy,  sy*sp],
            [-sp,      0,   cp   ],
        ])
        return pos_world, rot_world

    @staticmethod
    def world_to_base_frame(
        pos_world: np.ndarray,
        rot_world: np.ndarray,
        base_pos_world: np.ndarray,
        base_rot_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform a pose from world frame into the arm base (link00) frame."""
        R_bw = base_rot_world.T           # world→base rotation
        pos_base = R_bw @ (pos_world - base_pos_world)
        rot_base = R_bw @ rot_world
        return pos_base, rot_base

"""B2 locomotion controller using the ALORE low-level WBC policy."""

from typing import List

import numpy as np
import torch

import internutopia.core.util.math as math_utils
from internutopia.core.robot.articulation_action import ArticulationAction
from internutopia.core.robot.articulation_subset import ArticulationSubset
from internutopia.core.robot.controller import BaseController
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia_extension.configs.controllers import B2MoveBySpeedControllerCfg
from internutopia_extension.controllers.models.b2.low_level_model import ActorCriticLow

# ── B2Z1 joint ordering constants ──────────────────────────────────────────────
# Isaac Sim order (19 joints):
# [FL_hip, FR_hip, RL_hip, RR_hip,
#  FL_thigh, FR_thigh, RL_thigh, RR_thigh,
#  joint1, FL_calf, FR_calf, RL_calf, RR_calf,
#  joint2, joint3, joint4, joint5, joint6, jointGripper]

# Real robot order (19 joints):
# [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
#  RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf,
#  joint1, joint2, joint3, joint4, joint5, joint6, jointGripper]

# Reindex mappings (indices 0-based, same as ALORE _reindex_Isaacsim2real / _reindex_real2Isaacsim)
_SIM2REAL_IDX = [1, 5, 10, 0, 4, 9, 3, 7, 12, 2, 6, 11, 8]  # leg part (13 joints)
_REAL2SIM_IDX = [3, 0, 9, 6, 4, 1, 10, 7, 12, 5, 2, 11, 8]  # leg part (13 joints)
_LEG_CONTROL_IDX_SIM = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]

# Default joint positions in sim order (19 joints)
_DEFAULT_JOINT_POS_SIM = np.array([
    0.1, -0.1, 0.1, -0.1,       # FL_hip, FR_hip, RL_hip, RR_hip
    0.8, 0.8, 0.8, 0.8,          # FL_thigh, FR_thigh, RL_thigh, RR_thigh
    0.0, -1.5, -1.5, -1.5, -1.5, # joint1, FL_calf, FR_calf, RL_calf, RR_calf
    1.48, -0.63, -0.84, 0.0, 1.57, 0.0  # joint2..6, jointGripper
], dtype=np.float32)

# Action scale for low-level policy output (in sim order, 18 dims, no gripper)
_ACTION_SCALE_SIM = np.array([
    0.4, 0.4, 0.4, 0.4,          # FL_hip, FR_hip, RL_hip, RR_hip
    0.45, 0.45, 0.45, 0.45, 2.1, # FL_thigh, FR_thigh, RL_thigh, RR_thigh, joint1
    0.45, 0.45, 0.45, 0.45,       # FL_calf, FR_calf, RL_calf, RR_calf
    0.6, 0.6, 0.0, 0.0, 0.0      # joint2, joint3, joint4, joint5, joint6
], dtype=np.float32)

# Command scaling (match ALORE training: [2.0, 2.0, 0.25])
_COMMAND_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
_GAIT_COMMAND_THRESHOLD = 0.1
_DEFAULT_TARGET_WARMUP_CALLS = 10

# Low-level policy architecture params (must match trained checkpoint)
_LOW_LEVEL_KWARGS = {
    'continue_from_last_std': True,
    'init_std': [[0.8, 1.0, 1.0] * 4 + [1.0] * 6],
    'actor_hidden_dims': [512, 256, 128],
    'critic_hidden_dims': [512, 256, 128],
    'activation': 'elu',
    'output_tanh': False,
    'leg_control_head_hidden_dims': [128, 128],
    'arm_control_head_hidden_dims': [128, 128],
    'priv_encoder_dims': [64, 20],
    'num_leg_actions': 12,
    'num_arm_actions': 6,
    'adaptive_arm_gains': False,
    'adaptive_arm_gains_scale': 10.0,
}
_NUM_PROPRIO = 71
_NUM_PRIV = 18        # 5 + 1 + 12
_HISTORY_LEN = 10
_NUM_ACTIONS = 18
_PRIV_BUF = np.array([
    0.0000, 0.0000, 0.0000, 0.0000, 0.0795, 0.5203, -0.1516, -0.0065,
    0.0467, 0.2631, 0.1297, 0.1543, -0.1086, -0.1943, 0.0883, 0.2819,
    0.2323, -0.0110,
], dtype=np.float32)


def _sim2real(vec: np.ndarray) -> np.ndarray:
    """Reindex 19-element joint vector from Isaac Sim order to real robot order."""
    return np.concatenate([vec[_SIM2REAL_IDX], vec[13:]])


def _real2sim(vec: np.ndarray) -> np.ndarray:
    """Reindex 19-element joint vector from real robot order to Isaac Sim order."""
    return np.concatenate([vec[_REAL2SIM_IDX], vec[13:]])


def _quat2euler_rp(w: float, x: float, y: float, z: float):
    """Return (roll, pitch) from quaternion (w,x,y,z)."""
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(t0, t1))
    t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = float(np.arcsin(t2))
    return roll, pitch


@BaseController.register('B2MoveBySpeedController')
class B2MoveBySpeedController(BaseController):
    """Speed-command locomotion controller for Unitree B2Z1 using ALORE low-level WBC policy.

    Accepts a 3-element action [forward_speed, lateral_speed, rotation_speed] and
    drives the B2's 12 leg joints.  Arm joints are held at their default poses.
    """

    def __init__(self, config: B2MoveBySpeedControllerCfg, robot: BaseRobot, scene: IScene) -> None:
        super().__init__(config=config, robot=robot, scene=scene)

        self._policy = self._load_policy(config.policy_weights_path)
        self.joint_names = config.joint_names
        self.joint_subset: ArticulationSubset | None = None
        self.control_subset: ArticulationSubset | None = None
        if self.joint_names:
            self.joint_subset = ArticulationSubset(self.robot.articulation, self.joint_names)
            control_joint_names = [self.joint_names[i] for i in _LEG_CONTROL_IDX_SIM]
            self.control_subset = ArticulationSubset(self.robot.articulation, control_joint_names)

        # State buffers
        self._obs_history = np.zeros((_HISTORY_LEN, _NUM_PROPRIO), dtype=np.float32)
        self._history_initialized = False
        self._prev_action_sim = np.zeros(18, dtype=np.float32)
        self._gait_index = 0.0
        self._apply_times_left = 0
        self._applied_joint_positions: np.ndarray | None = None
        self._policy_calls = 0

    @staticmethod
    def _load_policy(path: str):
        model = ActorCriticLow(
            _NUM_PROPRIO,
            _NUM_PROPRIO,
            _NUM_ACTIONS,
            **_LOW_LEVEL_KWARGS,
            num_priv=_NUM_PRIV,
            num_hist=_HISTORY_LEN,
            num_prop=_NUM_PROPRIO,
        )
        ckpt = torch.load(path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model.act_inference

    def _step_gait(self, forward: float, lateral: float, rotation: float) -> tuple:
        """Advance gait phase; return (gait_index, clock_inputs[4])."""
        walking = (
            abs(forward) > _GAIT_COMMAND_THRESHOLD
            or abs(lateral) > _GAIT_COMMAND_THRESHOLD
            or abs(rotation) > _GAIT_COMMAND_THRESHOLD
        )
        if walking:
            self._gait_index = (self._gait_index + 0.02 * 2.0) % 1.0
        else:
            self._gait_index = 0.0

        phases, offsets, bounds = 0.5, 0.0, 0.0
        foot = [
            self._gait_index + phases + offsets + bounds,
            self._gait_index + offsets,
            self._gait_index + bounds,
            self._gait_index + phases,
        ]
        clock = np.array([np.sin(2 * np.pi * f) for f in foot], dtype=np.float32)
        return float(self._gait_index), clock

    def _build_obs(
        self,
        forward: float, lateral: float, rotation: float,
        joint_pos_sim: np.ndarray,
        joint_vel_sim: np.ndarray,
    ) -> np.ndarray:
        """Build the 71-dim per-step proprioceptive observation."""
        robot_base = self.robot.get_robot_base()
        pose = robot_base.get_pose()  # (position, orientation wxyz)
        quat = np.array(pose[1], dtype=np.float32)
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        # body orientation: roll, pitch
        roll, pitch = _quat2euler_rp(w, x, y, z)
        body_orientation = np.array([roll, pitch], dtype=np.float32)

        # angular velocity in body frame (InternUtopia math util)
        ang_vel_w = np.array(robot_base.get_angular_velocity(), dtype=np.float32)
        quat_t = torch.tensor(quat).reshape(1, 4)
        ang_vel_t = torch.tensor(ang_vel_w).reshape(1, 3)
        ang_vel_b = math_utils.quat_rotate_inverse(quat_t, ang_vel_t).numpy().reshape(3)
        ang_vel_scaled = (ang_vel_b * 0.25).astype(np.float32)

        # joint pos/vel: sim→real, remove gripper (last element → 18 remain)
        default_robot = _DEFAULT_JOINT_POS_SIM[:19]
        dof_pos_sim = joint_pos_sim - default_robot
        dof_pos_real = _sim2real(dof_pos_sim)[:-1]   # 18 dims
        dof_vel_real = _sim2real(joint_vel_sim * 0.05)[:-1]  # 18 dims

        # Previous low-level action, converted to real order and restricted to legs.
        prev_action_real_12 = _sim2real(self._prev_action_sim)[:12]

        # gait
        gait_index, clock = self._step_gait(forward, lateral, rotation)

        # ee_goal_local: set to a neutral held pose (arm fully folded toward back)
        ee_goal_local = np.zeros(3, dtype=np.float32)

        command = np.array([forward, lateral, rotation], dtype=np.float32) * _COMMAND_SCALE
        zeros4 = np.zeros(4, dtype=np.float32)
        zeros3 = np.zeros(3, dtype=np.float32)

        obs = np.concatenate([
            body_orientation,    # 2
            ang_vel_scaled,      # 3
            dof_pos_real,        # 18
            dof_vel_real,        # 18
            prev_action_real_12, # 12
            zeros4,              # 4  (contact targets)
            command,             # 3
            ee_goal_local,       # 3
            zeros3,              # 3
            [gait_index],        # 1
            clock,               # 4
        ]).astype(np.float32)
        # total = 2+3+18+18+12+4+3+3+3+1+4 = 71
        return obs

    def forward(
        self,
        forward_speed: float = 0.0,
        lateral_speed: float = 0.0,
        rotation_speed: float = 0.0,
    ) -> ArticulationAction:
        if self._apply_times_left > 0:
            self._apply_times_left -= 1
            if self.control_subset is not None:
                return self.control_subset.make_articulation_action(
                    joint_positions=self._applied_joint_positions, joint_velocities=None
                )
            if self.joint_subset is not None:
                return self.joint_subset.make_articulation_action(
                    joint_positions=self._applied_joint_positions, joint_velocities=None
                )
            else:
                return ArticulationAction(joint_positions=self._applied_joint_positions)

        # Get current joint state (all 19 joints in Isaac Sim order)
        if self.joint_subset is not None:
            joint_pos_sim = self.joint_subset.get_joint_positions()
            joint_vel_sim = self.joint_subset.get_joint_velocities()
        else:
            joint_pos_sim = self.robot.articulation.get_joint_positions()
            joint_vel_sim = self.robot.articulation.get_joint_velocities()

        obs = self._build_obs(forward_speed, lateral_speed, rotation_speed,
                              joint_pos_sim, joint_vel_sim)
        if not self._history_initialized:
            self._obs_history[:] = obs
            self._history_initialized = True

        # Match ALORE: policy sees current obs plus the previous history buffer.
        priv_buf = _PRIV_BUF
        history_flat = self._obs_history.flatten()
        policy_input = np.concatenate([obs, priv_buf, history_flat]).reshape(1, -1)

        self._policy_calls += 1
        if self._policy_calls <= _DEFAULT_TARGET_WARMUP_CALLS:
            action_real = np.zeros(18, dtype=np.float32)
        else:
            # Infer
            with torch.inference_mode():
                action_real = (
                    self._policy(
                        torch.tensor(policy_input, dtype=torch.float32),
                        hist_encoding=True,
                    )
                    .detach()
                    .numpy()[0]
            )  # shape (18,) in real order

        self._obs_history = np.roll(self._obs_history, -1, axis=0)
        self._obs_history[-1] = obs

        # Zero arm joints (high-level arm control handles those separately)
        action_real[12:] = 0.0

        # Reindex to Isaac Sim order (18 controlled joints, no gripper).
        action_sim_18 = _real2sim(action_real)[:18]

        self._prev_action_sim = action_sim_18.copy()

        # Scale and add default pose
        joint_targets = _ACTION_SCALE_SIM * action_sim_18 + _DEFAULT_JOINT_POS_SIM[:18]

        # Pad gripper back to 19 joints, keep gripper at default
        full_targets = np.concatenate([joint_targets, [_DEFAULT_JOINT_POS_SIM[18]]])
        if self.control_subset is not None:
            self._applied_joint_positions = full_targets[_LEG_CONTROL_IDX_SIM]
        else:
            self._applied_joint_positions = full_targets
        self._apply_times_left = 3  # apply same action for 4 sim steps (like ALORE decimation=4)

        if self.control_subset is not None:
            return self.control_subset.make_articulation_action(
                joint_positions=self._applied_joint_positions, joint_velocities=None
            )
        if self.joint_subset is not None:
            return self.joint_subset.make_articulation_action(
                joint_positions=self._applied_joint_positions, joint_velocities=None
            )
        return ArticulationAction(joint_positions=self._applied_joint_positions)

    def action_to_control(self, action: list | np.ndarray) -> ArticulationAction:
        """Convert [forward_speed, lateral_speed, rotation_speed] to joint positions."""
        assert len(action) == 3, 'B2MoveBySpeedController action must have 3 elements'
        return self.forward(
            forward_speed=float(action[0]),
            lateral_speed=float(action[1]),
            rotation_speed=float(action[2]),
        )

    def get_obs(self) -> dict:
        return {}

from pathlib import Path
from typing import List, Optional, Tuple

from internutopia.core.config import RobotCfg
from internutopia_extension.configs.controllers import B2MoveBySpeedControllerCfg, JointControllerCfg
from internutopia_extension.configs.controllers.arm_ik_controller import ArmIKControllerCfg

_REPO_ROOT = Path(__file__).resolve().parents[3]
B2Z1_ASSET_DIR = _REPO_ROOT / 'local_assets' / 'robots' / 'b2z1'
B2Z1_USD_PATH = str(B2Z1_ASSET_DIR / 'urdf' / 'b2z1_nolidar' / 'b2z1_nolidar.usd')

# Default locomotion policy copied from ALORE into this repo.
B2Z1_LOCO_POLICY_PATH = str(_REPO_ROOT / 'local_assets' / 'policies' / 'b2z1' / 'model_57000.pt')

# Isaac Sim joint order for B2Z1 (19 joints total)
B2Z1_LEG_JOINTS: List[str] = [
    'FL_hip_joint',
    'FR_hip_joint',
    'RL_hip_joint',
    'RR_hip_joint',
    'FL_thigh_joint',
    'FR_thigh_joint',
    'RL_thigh_joint',
    'RR_thigh_joint',
    'joint1',        # Z1 arm joint1 (interleaved in URDF)
    'FL_calf_joint',
    'FR_calf_joint',
    'RL_calf_joint',
    'RR_calf_joint',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'jointGripper',
]

B2Z1_ARM_JOINTS: List[str] = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'jointGripper',
]

move_by_speed_cfg = B2MoveBySpeedControllerCfg(
    name='move_by_speed',
    policy_weights_path=B2Z1_LOCO_POLICY_PATH,
    joint_names=B2Z1_LEG_JOINTS,
)

arm_controller_cfg = JointControllerCfg(
    name='arm_control',
    joint_names=B2Z1_ARM_JOINTS,
)

arm_ik_controller_cfg = ArmIKControllerCfg(
    name='arm_ik',
    joint_names=B2Z1_ARM_JOINTS,
    arm_base_prim_suffix='/link00',
)


class B2Z1RobotCfg(RobotCfg):
    name: Optional[str] = 'b2z1'
    type: Optional[str] = 'B2Z1Robot'
    prim_path: Optional[str] = '/b2z1'
    usd_path: Optional[str] = B2Z1_USD_PATH
    base_link_suffixes: Tuple[str, ...] = (
        '/base',
        '/trunk',
        '/b2/base',
        '/b2/trunk',
    )
    # arm_control: direct joint positions; arm_ik: world-frame EE target via Pinocchio IK
    controllers: List = [move_by_speed_cfg, arm_controller_cfg, arm_ik_controller_cfg]

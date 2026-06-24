from typing import List, Optional

from internutopia.core.config.robot import ControllerCfg


class ArmIKControllerCfg(ControllerCfg):
    """Controller that accepts a world-frame EE target pose and drives the Z1 arm via IK."""

    type: Optional[str] = 'ArmIKController'
    joint_names: List[str]        # arm joints in Isaac Sim order
    arm_base_prim_suffix: str = '/link00'  # prim path suffix under robot prim_path
    urdf_path: Optional[str] = None       # Z1 URDF path; if None uses default

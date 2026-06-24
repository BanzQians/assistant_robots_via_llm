from typing import List, Optional

from internutopia.core.config.robot import ControllerCfg


class B2MoveBySpeedControllerCfg(ControllerCfg):

    type: Optional[str] = 'B2MoveBySpeedController'
    joint_names: List[str]
    policy_weights_path: str

"""Demo: load B2Z1 in InternUtopia and hold a fixed standing pose.

This intentionally does not use the ALORE locomotion policy. It is a visual
asset/physics smoke test for the B2Z1 USD in InternUtopia.
"""

import argparse

import numpy as np

from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia.macros import gm
from internutopia_extension import import_extensions
from internutopia_extension.configs.controllers import JointControllerCfg
from internutopia_extension.configs.robots.b2z1 import B2Z1_LEG_JOINTS, B2Z1RobotCfg
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg

B2Z1_HOME = np.array([
    0.1, -0.1, 0.1, -0.1,
    0.8, 0.8, 0.8, 0.8,
    0.0, -1.5, -1.5, -1.5, -1.5,
    1.48, -0.63, -0.84, 0.0, 1.57, -1.2,
], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description='Load B2Z1 and hold a fixed standing pose.')
    parser.add_argument('--steps', type=int, default=0, help='0 = run until window is closed.')
    parser.add_argument('--headless', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    headless = args.headless or not has_display()

    hold_controller = JointControllerCfg(name='joint_hold', joint_names=B2Z1_LEG_JOINTS)
    config = Config(
        simulator=SimConfig(
            physics_dt=1 / 200,
            rendering_dt=1 / 50,
            use_fabric=False,
            headless=headless,
            webrtc=headless,
        ),
        task_configs=[
            SingleInferenceTaskCfg(
                scene_asset_path=gm.ASSET_PATH + '/scenes/empty.usd',
                robots=[
                    B2Z1RobotCfg(
                        position=(0.0, 0.0, 0.55),
                        controllers=[hold_controller],
                    )
                ],
            )
        ],
    )

    import_extensions()
    env = Env(config)
    obs, _ = env.reset()

    print('=== B2Z1 Asset Load Demo ===')
    print(f'  Robot start : {np.round(obs["position"], 3)}')

    step = 0
    try:
        while env.simulation_app.is_running():
            step += 1
            obs, _, terminated, _, _ = env.step(action={'joint_hold': [B2Z1_HOME]})

            if step % 200 == 0:
                print(f'  step {step:5d}: pos={np.round(obs["position"], 3)}')

            if (args.steps > 0 and step >= args.steps) or terminated:
                break
    finally:
        print('Demo finished.')
        env.close()


if __name__ == '__main__':
    main()

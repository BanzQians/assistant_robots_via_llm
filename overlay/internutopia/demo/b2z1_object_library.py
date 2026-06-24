"""B2Z1 物体抓取参数库 —— 按类别（category）定义可抓取物体。

每个类别包含：
    usd        : USD 资产路径
    grasp_cfg  : (dx, dz, pitch_deg) 抓取点相对物体中心的偏移
                 dx = 机械臂从物体正前方接近的水平距离 [m]
                 dz = 抓取点相对物体中心的高度偏移 [m]
                 pitch_deg = 夹爪俯仰角 [度]
    plan_dist  : 机器人停在距物体多远处开始抓 [m]
    com_offset : (dx, dy) 物体质心相对几何中心的偏移补偿 [m]
    grip_close : 夹爪闭合到的角度 [rad]，越接近 0 越闭合（物体越大越不能完全闭合）

这些参数来自 ALORE config.yaml，可根据实际测试结果调整。
"""

import os

_THIS = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, '../..'))
_ASSETS = os.path.join(_REPO_ROOT, 'local_assets', 'objects', 'alore')


OBJECT_LIBRARY = {
    'box': {
        'usd':        os.path.join(_ASSETS, 'object_7', 'box.usd'),
        'grasp_cfg':  (0.25, 0.25, 80.0),
        'plan_dist':  0.83,
        'com_offset': (0.0, 0.0),
        'grip_close': -0.1,
    },
    'chair': {
        'usd':        os.path.join(_ASSETS, 'object_1', 'model_office_chair_3_v1.usd'),
        'grasp_cfg':  (0.45, 0.80, 60.0),
        'plan_dist':  0.80,
        'com_offset': (0.0, 0.0),
        'grip_close': -0.3,
    },
    'table': {
        'usd':        os.path.join(_ASSETS, 'object_4', 'table4_2.usd'),
        'grasp_cfg':  (0.50, 0.62, 6.0),
        'plan_dist':  1.07,
        'com_offset': (0.3, 0.5),
        'grip_close': -0.4,
    },
}


def get_object_params(category: str) -> dict:
    """返回某类别的抓取参数；未知类别抛出明确错误。"""
    if category not in OBJECT_LIBRARY:
        raise ValueError(
            f'未知物体类别 "{category}"。可用类别: {list(OBJECT_LIBRARY.keys())}'
        )
    return OBJECT_LIBRARY[category]


def list_categories() -> list:
    return list(OBJECT_LIBRARY.keys())

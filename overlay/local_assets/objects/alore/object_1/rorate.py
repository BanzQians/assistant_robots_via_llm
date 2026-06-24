from pxr import Usd, UsdGeom, Gf

stage = Usd.Stage.Open("/home/v1/IsaacLab/assets/movechair2/model_office_chair_3_v1.usd")

# 获取原始根层内容（你想包裹起来的节点）
old_root = stage.GetPrimAtPath("/Root")  # 替换为你的主节点路径

# 创建一个新的 transform 作为父级
transform_prim = UsdGeom.Xform.Define(stage, "/RotatedRoot")

# 设置绕 Z 轴旋转 -90 度
xform = UsdGeom.Xform(transform_prim)
xform.AddRotateZOp().Set(-90)

# 将原始节点设为子节点
stage.GetRootLayer().ImportPrim(old_root.GetPath(), "/RotatedRoot/Root")

# 保存
stage.GetRootLayer().Save()


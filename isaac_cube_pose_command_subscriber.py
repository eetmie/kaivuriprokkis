"""Isaac Sim Python Script node: move the red cube from ROS pose commands.

This Script Node subscribes to /kaivuri/cube_pose_cmd and moves
/World/red_cube.

The incoming command is in the excavator frame and uses the same convention as
cube_touch_expert_node: the position is the cube top-center in the IK frame.
"""

import omni
import rclpy
from geometry_msgs.msg import PoseStamped
from pxr import Gf, UsdGeom


CUBE_PRIM_PATH = "/World/red_cube"
EXCAVATOR_PRIM_PATH = "/World/excavator"
TOPIC = "/kaivuri/cube_pose_cmd"
FRAME_ID = "excavator"
IK_ORIGIN_IN_EXCAVATOR_FRAME = Gf.Vec3d(0.0, 0.0, -0.05042)


def _set_translate(prim, xyz, gf=Gf, usd_geom=UsdGeom):
    xform = usd_geom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == usd_geom.XformOp.TypeTranslate:
            op.Set(gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
            return
    xform.AddTranslateOp().Set(gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))


def _on_cube_pose_cmd(
    msg,
    cube_prim_path=CUBE_PRIM_PATH,
    excavator_prim_path=EXCAVATOR_PRIM_PATH,
    frame_id=FRAME_ID,
    ik_origin_in_excavator_frame=IK_ORIGIN_IN_EXCAVATOR_FRAME,
    omni_module=omni,
    gf=Gf,
    usd_geom=UsdGeom,
    set_translate=_set_translate,
):
    stage = omni_module.usd.get_context().get_stage()
    timeline = omni_module.timeline.get_timeline_interface()
    time_code = timeline.get_current_time()

    cube_prim = stage.GetPrimAtPath(cube_prim_path)
    excavator_prim = stage.GetPrimAtPath(excavator_prim_path)
    if not cube_prim.IsValid() or not excavator_prim.IsValid():
        print(f"Cube command ignored: missing {cube_prim_path} or {excavator_prim_path}")
        return

    if msg.header.frame_id and msg.header.frame_id != frame_id:
        print(f"Cube command ignored: expected frame {frame_id}, got {msg.header.frame_id}")
        return

    # The publisher sends:
    #   top_center_ik = top_center_excavator - IK_ORIGIN_IN_EXCAVATOR_FRAME
    # Invert that here. Then move the cube by bbox-top-center delta.
    desired_top_center_ik = gf.Vec3d(
        float(msg.pose.position.x),
        float(msg.pose.position.y),
        float(msg.pose.position.z),
    )
    desired_top_center_excavator = desired_top_center_ik + ik_origin_in_excavator_frame

    excavator_tf = usd_geom.Xformable(excavator_prim).ComputeLocalToWorldTransform(time_code)
    desired_top_center_world = excavator_tf.Transform(desired_top_center_excavator)

    cache = usd_geom.BBoxCache(time_code, [usd_geom.Tokens.default_, usd_geom.Tokens.render])
    cube_bbox = cache.ComputeWorldBound(cube_prim).ComputeAlignedBox()
    mn = cube_bbox.GetMin()
    mx = cube_bbox.GetMax()
    current_top_center_world = gf.Vec3d(
        (mn[0] + mx[0]) * 0.5,
        (mn[1] + mx[1]) * 0.5,
        mx[2],
    )
    delta_world = desired_top_center_world - current_top_center_world

    cube_xform = usd_geom.Xformable(cube_prim)
    cube_tf = cube_xform.ComputeLocalToWorldTransform(time_code)
    current_origin_world = cube_tf.Transform(gf.Vec3d(0.0, 0.0, 0.0))
    new_origin_world = current_origin_world + delta_world

    set_translate(cube_prim, new_origin_world)
    print(f"Moved red cube top center to {desired_top_center_world}")


if not hasattr(db.per_instance_state, "cube_pose_cmd_initialized"):
    if not rclpy.ok():
        rclpy.init()

    db.per_instance_state.cube_pose_cmd_node = rclpy.create_node("isaac_cube_pose_command_subscriber")
    db.per_instance_state.cube_pose_cmd_sub = db.per_instance_state.cube_pose_cmd_node.create_subscription(
        PoseStamped,
        TOPIC,
        _on_cube_pose_cmd,
        10,
    )
    db.per_instance_state.cube_pose_cmd_initialized = True
    print(f"Initialized cube command subscriber on {TOPIC}")

rclpy.spin_once(db.per_instance_state.cube_pose_cmd_node, timeout_sec=0.0)

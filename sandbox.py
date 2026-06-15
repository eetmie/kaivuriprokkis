import omni
import rclpy
from geometry_msgs.msg import PoseStamped
from pxr import UsdGeom, Gf

CUBE_PRIM_PATH = "/World/red_cube"
EXCAVATOR_PRIM_PATH = "/World/excavator"
TOPIC = "/kaivuri/cube_pose"
FRAME_ID = "excavator"

if not hasattr(db.per_instance_state, "initialized"):
    if not rclpy.ok():
        rclpy.init()

    db.per_instance_state.node = rclpy.create_node("isaac_cube_top_pose_publisher")
    db.per_instance_state.pub = db.per_instance_state.node.create_publisher(PoseStamped, TOPIC, 10)
    db.per_instance_state.initialized = True
    print("Initialized cube top pose publisher")

stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()

cube_prim = stage.GetPrimAtPath(CUBE_PRIM_PATH)
excavator_prim = stage.GetPrimAtPath(EXCAVATOR_PRIM_PATH)

if cube_prim.IsValid() and excavator_prim.IsValid():
    time_code = timeline.get_current_time()

    cache = UsdGeom.BBoxCache(
        time_code,
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )

    cube_bbox = cache.ComputeWorldBound(cube_prim).ComputeAlignedBox()
    mn = cube_bbox.GetMin()
    mx = cube_bbox.GetMax()

    top_center_world = Gf.Vec3d(
        (mn[0] + mx[0]) * 0.5,
        (mn[1] + mx[1]) * 0.5,
        mx[2],
    )
    print(f"Top center in world coordinates: {top_center_world}")
    excavator_tf = UsdGeom.Xformable(excavator_prim).ComputeLocalToWorldTransform(time_code)
    IK_FRAME_Z_BELOW_EXCAVATOR = 0.09
    top_center_excavator = excavator_tf.GetInverse().Transform(top_center_world)
    print(f"Top center in excavator coordinates: {top_center_excavator}")

    msg = PoseStamped()
    msg.header.stamp = db.per_instance_state.node.get_clock().now().to_msg()
    msg.header.frame_id = FRAME_ID
    msg.pose.position.x  = float(top_center_excavator[0])
    msg.pose.position.y  = float(top_center_excavator[1])
    msg.pose.position.z  = float(top_center_excavator[2] + IK_FRAME_Z_BELOW_EXCAVATOR)
    msg.pose.orientation.w = 1.0
    print(f"Publishing cube top pose: {msg}")
    db.per_instance_state.pub.publish(msg)

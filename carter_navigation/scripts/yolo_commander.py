#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from rclpy.time import Time
import math 

class YoloCommander(Node):
    def __init__(self):
        super().__init__('yolo_commander_node')
        
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt') 
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.current_goal_handle = None 

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        goal_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.rgb_topic = '/front_stereo_camera/left/image_raw'
        self.depth_topic = '/front_stereo_camera/right/image_depth'
        
        self.create_subscription(Image, self.rgb_topic, self.rgb_callback, qos_profile=sensor_qos)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile=sensor_qos)
        self.create_subscription(PoseStamped, '/rviz_goal_pose', self.rviz_goal_callback, qos_profile=goal_qos)

        self.pub_status = self.create_publisher(String, '/yolo_status', 10)
        self.pub_yolo_img = self.create_publisher(Image, '/yolo_image', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_global_plan = self.create_publisher(Path, '/plan', 10)
        self.pub_path = self.create_publisher(Path, '/robot_path', 10)
        
        self.history_path = Path()
        self.history_path.header.frame_id = 'map'
        self.create_timer(0.2, self.update_path_callback)

        self.latest_depth = None
        self.mode = "IDLE" 
        
        self.fx = 640.0
        self.fy = 480.0
        self.cx = 640.0 / 2
        self.cy = 480.0 / 2

        self.bag_detection_count = 0        
        self.bag_detection_threshold = 1 
        self.person_max_detection_range = 3.0 

        self.get_logger().info(">>> YOLO Commander Ready (Go Left 4.0m)")

    def update_path_callback(self):
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time(seconds=0))
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            self.history_path.header.stamp = self.get_clock().now().to_msg()
            self.history_path.poses.append(pose)
            if len(self.history_path.poses) > 5000: self.history_path.poses.pop(0)
            self.pub_path.publish(self.history_path)
        except: pass

    def depth_callback(self, msg):
        try: self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except: pass

    def rviz_goal_callback(self, msg):
        self.get_logger().info(">>> [USER CMD] Received Goal from RViz! Navigating...")
        self.mode = "NAVIGATING"
        self.bag_detection_count = 0
        self.send_nav_goal(msg)

    def send_nav_goal(self, pose_stamped, callback_type="DEFAULT"):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_stamped
        
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(">>> [ERROR] Nav2 Action Server time out!")
            return
        
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            return
        self.current_goal_handle = goal_handle

    def stop_robot(self):
        stop_msg = Twist()
        self.pub_cmd_vel.publish(stop_msg)
        if self.current_goal_handle and self.current_goal_handle.status == GoalStatus.STATUS_EXECUTING:
            self.current_goal_handle.cancel_goal_async()
            self.current_goal_handle = None
        
        empty_path = Path()
        empty_path.header.frame_id = 'map'
        empty_path.header.stamp = self.get_clock().now().to_msg()
        self.pub_global_plan.publish(empty_path)

    def get_distance_to_center(self, x1, y1, x2, y2):
        if self.latest_depth is None: return 0.0
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        h, w = self.latest_depth.shape
        if not (0 <= cx < w and 0 <= cy < h): return 0.0
        roi = self.latest_depth[max(0, cy-5):min(h, cy+5), max(0, cx-5):min(w, cx+5)]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        return float(np.median(valid)) if len(valid) > 0 else 0.0

    def calculate_map_pose(self, x1, x2, y1, y2, dist, header_rgb):
        stop_offset = 0.25 
        if dist <= stop_offset: return None
        try:
            z_target = dist - stop_offset
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            x_cam = (u - self.cx) * z_target / self.fx
            y_cam = (v - self.cy) * z_target / self.fy
            z_cam = z_target

            point_cam = PoseStamped()
            point_cam.header = header_rgb
            if point_cam.header.frame_id.startswith('/'):
                point_cam.header.frame_id = point_cam.header.frame_id[1:]
            point_cam.header.stamp = rclpy.time.Time(seconds=0).to_msg()
            point_cam.pose.position.x = x_cam
            point_cam.pose.position.y = y_cam
            point_cam.pose.position.z = z_cam
            point_cam.pose.orientation.w = 1.0
            
            target_pose = self.tf_buffer.transform(
                point_cam, 'map', timeout=rclpy.duration.Duration(seconds=0.1)
            )
            target_pose.pose.position.z = 0.0 
            return target_pose
        except: return None

    # [핵심 변경] 왼쪽으로 4.0m 이동하는 좌표 계산 함수
    def get_left_goal_pose(self):
        try:
            left_pose = PoseStamped()
            left_pose.header.frame_id = 'base_link'
            left_pose.header.stamp = self.get_clock().now().to_msg()
            
            left_pose.pose.position.x = 0.5 
            # [수정] 왼쪽으로 4.0m 이동! (기존 2.0 -> 4.0)
            left_pose.pose.position.y = 4.0  
            left_pose.pose.position.z = 0.0
            
            left_pose.pose.orientation.z = 0.707
            left_pose.pose.orientation.w = 0.707
            
            target_pose = self.tf_buffer.transform(
                left_pose, 
                'map', 
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            target_pose.pose.position.z = 0.0
            return target_pose
            
        except Exception as e:
            self.get_logger().error(f"Failed to calc left pose: {e}")
            return None

    def rgb_callback(self, msg):
        try: frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except: return

        results = self.model(frame, classes=[0, 24, 26, 28], verbose=False, conf=0.16)
        
        person_info = None
        bag_info = None
        
        if any(len(r.boxes) > 0 for r in results):
            for r in results:
                for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                    cls_id = int(cls)
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2-x1) * (y2-y1)
                    dist = self.get_distance_to_center(x1, y1, x2, y2)

                    if cls_id == 0: 
                        if person_info is None or area > person_info['area']:
                            person_info = {'box': (x1, y1, x2, y2), 'dist': dist, 'area': area}
                    elif cls_id in [24, 26, 28]: 
                        if bag_info is None or area > bag_info['area']:
                            bag_info = {'box': (x1, y1, x2, y2), 'dist': dist, 'area': area, 'cls': cls_id}

        final_target = None
        target_type = "NONE"

        # 우선순위: 가방 > 사람
        if bag_info:
            final_target = bag_info
            target_type = "BAG"
            self.bag_detection_count += 1
        elif person_info:
            final_target = person_info
            target_type = "PERSON"
            self.bag_detection_count = 0 
        else:
            self.bag_detection_count = 0

        if final_target:
            x1, y1, x2, y2 = final_target['box']
            color = (0, 255, 0) if target_type == "BAG" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        try: self.pub_yolo_img.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))
        except: pass

        if final_target and final_target['dist'] > 0:
            dist = final_target['dist']

            # [CASE 1: 가방] 발견 즉시 왼쪽으로 4m 도망가기
            if target_type == "BAG":
                if self.mode == "NAVIGATING":
                    if self.bag_detection_count >= self.bag_detection_threshold:
                        self.get_logger().warn(f">>> 🎒 BAG Detected! GO LEFT 4.0m!")
                        self.stop_robot() 
                        
                        left_goal = self.get_left_goal_pose()
                        
                        if left_goal:
                            self.mode = "MOVING_LEFT"
                            self.send_nav_goal(left_goal, callback_type="LEFT")
                        else:
                            self.mode = "NAVIGATING"

            # [CASE 2: 사람]
            elif target_type == "PERSON":
                if self.mode != "IDLE" and self.mode != "MOVING_LEFT":
                    if dist <= self.person_max_detection_range:
                        if dist > 0.3: 
                            x1, y1, x2, y2 = final_target['box']
                            target_pose = self.calculate_map_pose(x1, x2, y1, y2, dist, msg.header)

                            if target_pose:
                                self.mode = "APPROACHING_PERSON"
                                self.send_nav_goal(target_pose, callback_type="PERSON")
                        else:
                            if self.mode == "APPROACHING_PERSON" and dist <= 0.3:
                                self.stop_robot()
                                self.mode = "IDLE"

        status_msg = String()
        status_msg.data = f"Mode: {self.mode}"
        self.pub_status.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
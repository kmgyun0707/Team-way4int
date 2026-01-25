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
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from rclpy.time import Time
import traceback

# 시나리오1 최종안

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
        self.publisher_rviz = self.create_publisher(Image, 'person_detected_view', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_global_plan = self.create_publisher(Path, '/plan', 10)
        self.pub_path = self.create_publisher(Path, '/robot_path', 10)
        
        self.history_path = Path()
        self.history_path.header.frame_id = 'map'
        self.create_timer(0.2, self.update_path_callback)

        self.latest_depth = None
        self.mode = "IDLE"
        self.last_person_pose = None
        
        # 쿨타임 관리
        self.last_goal_time = self.get_clock().now()
        
        self.fx = 640.0
        self.fy = 480.0
        self.cx = 640.0 / 2
        self.cy = 480.0 / 2

        self.get_logger().info(">>> YOLO Commander Ready (Smooth Transition Mode)")

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
        self.get_logger().info(">>> [USER CMD] Received Goal from RViz!")
        self.mode = "NAVIGATING"
        self.send_nav_goal(msg)

    def send_nav_goal(self, pose_stamped):
        # FOLLOWING 모드일 때만 쿨타임 적용 (사람 따라갈 때 버벅임 방지)
        if self.mode == "FOLLOWING":
            current_time = self.get_clock().now()
            time_diff = (current_time - self.last_goal_time).nanoseconds / 1e9
            # 0.8초 쿨타임 (너무 자주 보내면 Nav2가 경로짜다 포기함)
            if time_diff < 0.8: 
                return 

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_stamped
        
        if not self.nav_client.wait_for_server(timeout_sec=0.5): return
        
        self.get_logger().info(f">>> [MOVE] Sending Goal... (Dist: {pose_stamped.pose.position.x:.1f})")
        
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)
        self.last_goal_time = self.get_clock().now()

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            # 여기가 중요: Nav2가 "못 가!" 하고 거절한 경우
            self.get_logger().warn(">>> [WARN] Nav2 Rejected Goal! (Target is inside Obstacle?)")
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

    def rgb_callback(self, msg):
        status_msg = String()
        try: frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except: return

        results = self.model(frame, classes=[0], verbose=False, conf=0.6)
        
        best_box = None
        max_area = 0
        if any(len(r.boxes) > 0 for r in results):
            for r in results:
                for box in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2-x1) * (y2-y1)
                    if area > max_area:
                        max_area = area
                        best_box = (x1, y1, x2, y2)

        # 1. 화면 그리기 (최우선)
        dist = 0.0
        if best_box:
            x1, y1, x2, y2 = best_box
            dist = self.get_distance_to_center(x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"TARGET {dist:.1f}m", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 2. 이미지 송출
        try: 
            ros_img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            ros_img_msg.header = msg.header
            self.pub_yolo_img.publish(ros_img_msg)
        except: pass

        # 3. 행동 로직 (끊김 없는 전환)
        if best_box and dist > 0:
            
            # [상황 1] 주행 중 사람 발견 -> 정지 후 바로 추적 모드로 진입 (Return 안 함!)
            if self.mode == "NAVIGATING":
                self.get_logger().warn(f">>> 🚨 Person Detected! Stopping & Approaching immediately.")
                self.stop_robot() 
                self.mode = "FOLLOWING"
                # 주의: 여기서 return을 제거했습니다. 
                # 바로 아래의 FOLLOWING 로직으로 이어져서 즉시 이동 명령을 계산합니다.

            # [상황 2] dist 는 1.3, 0.35m 여유 두고 접근 이게 가장 이상적인게 한번씩 나오긴함
            if self.mode == "FOLLOWING":
                if dist > 1.3: 
                    # 목표 지점을 조금 더 여유 있게 잡음 (충돌 방지)
                    # dist - 0.5가 너무 가까우면 Nav2가 거부하므로
                    # 상황에 따라 0.6~0.7로 늘려주는 것도 방법
                    target_dist = dist - 0.35
                    if target_dist < 0: target_dist = 0.0
                    
                    target_pose = self.calculate_map_pose(x1, x2, y1, y2, target_dist, msg.header)
                    
                    if target_pose:
                        self.send_nav_goal(target_pose)
                        status_msg.data = f"Go: {target_dist:.1f}m"
                    else:
                        status_msg.data = "TF Error"
                else:
                    self.stop_robot()
                    status_msg.data = f"🛑 Arrived ({dist:.1f}m)"
                    
        else:
            # 사람 놓침
            if self.mode == "FOLLOWING":
                self.stop_robot()
                self.mode = "IDLE"
                status_msg.data = "Lost Person"
            elif self.mode == "NAVIGATING":
                status_msg.data = "Moving to Goal..."
            else:
                status_msg.data = "IDLE"

        self.pub_status.publish(status_msg)

    def get_distance_to_center(self, x1, y1, x2, y2):
        if self.latest_depth is None: return 0.0
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        h, w = self.latest_depth.shape
        if not (0 <= cx < w and 0 <= cy < h): return 0.0
        roi = self.latest_depth[max(0, cy-5):min(h, cy+5), max(0, cx-5):min(w, cx+5)]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        return float(np.median(valid)) if len(valid) > 0 else 0.0

    def calculate_map_pose(self, x1, x2, y1, y2, dist, header_rgb):
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        z_cam = dist
        x_cam = (u - self.cx) * z_cam / self.fx
        y_cam = (v - self.cy) * z_cam / self.fy
        point_cam = PoseStamped()
        point_cam.header = header_rgb
        point_cam.header.frame_id = 'front_stereo_camera_left_rgb'
        point_cam.header.stamp = rclpy.time.Time(seconds=0).to_msg()
        point_cam.pose.position.x = x_cam
        point_cam.pose.position.y = y_cam
        point_cam.pose.position.z = z_cam
        point_cam.pose.orientation.w = 1.0
        try:
            target_pose = self.tf_buffer.transform(
                point_cam, 
                'map', 
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            target_pose.pose.position.z = 0.0 
            target_pose.header.stamp = self.get_clock().now().to_msg()
            return target_pose
        except: return None

def main(args=None):
    rclpy.init(args=args)
    node = YoloCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
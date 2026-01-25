#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
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

        qos_policy = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        # 토픽 설정
        self.rgb_topic = '/front_stereo_camera/left/image_raw'
        self.depth_topic = '/front_stereo_camera/right/image_depth'
        
        self.create_subscription(Image, self.rgb_topic, self.rgb_callback, qos_profile=qos_policy)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile=qos_policy)
        
        # RViz에서 '2D Goal Pose'를 찍으면 이리로 들어옵니다.
        self.create_subscription(PoseStamped, '/rviz_goal_pose', self.rviz_goal_callback, 10)

        self.pub_status = self.create_publisher(String, '/yolo_status', 10)
        self.publisher_rviz = self.create_publisher(Image, 'person_detected_view', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_global_plan = self.create_publisher(Path, '/plan', 10)

        self.latest_depth = None
        self.mode = "IDLE" # 상태: IDLE, NAVIGATING(RViz목표), FOLLOWING(사람추적)
        self.last_person_pose = None
        
        self.fx = 640.0
        self.fy = 480.0
        self.cx = 640.0 / 2
        self.cy = 480.0 / 2

        self.get_logger().info(">>> YOLO Commander: Priority Mode (Person > RViz Goal)")

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except: pass

    # 1. 사용자가 RViz에서 목표를 찍었을 때
    def rviz_goal_callback(self, msg):
        self.get_logger().info(">>> [USER CMD] RViz Goal Received! Moving...")
        self.mode = "NAVIGATING" # 모드 변경
        self.is_moving_to_person = False
        
        # Nav2로 목표 전송 (우리가 직접 안 보내고, Nav2의 Goal Pose 토픽을 채가서 처리하거나 직접 액션 전송)
        # /goal_pose를 구독했으므로, 여기서 직접 Nav2 Action을 쏴줍니다.
        self.send_nav_goal(msg)

    def send_nav_goal(self, pose_stamped):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_stamped
        
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            return

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if goal_handle.accepted:
            self.current_goal_handle = goal_handle

    def stop_robot(self):
        """로봇 강제 정지 및 경로 삭제"""
        stop_msg = Twist()
        self.pub_cmd_vel.publish(stop_msg)
        
        if self.current_goal_handle and self.current_goal_handle.status == GoalStatus.STATUS_EXECUTING:
            self.current_goal_handle.cancel_goal_async()
        
        # RViz 경로선 지우기
        empty_path = Path()
        empty_path.header.frame_id = 'map'
        empty_path.header.stamp = self.get_clock().now().to_msg()
        self.pub_global_plan.publish(empty_path)

    def rgb_callback(self, msg):
        status_msg = String()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except: return

        # YOLO 추론
        results = self.model(frame, classes=[0], verbose=False, conf=0.6)
        person_detected = False
        
        # 사람 찾기
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
        
        if best_box:
            person_detected = True
            x1, y1, x2, y2 = best_box
            dist = self.get_distance_to_center(x1, y1, x2, y2)

            # ==========================================
            # 🔥 핵심 로직: 인터럽트 발생!
            # ==========================================
            if dist > 0: # 유효한 거리라면
                
                # 시나리오: RViz 찍고 가고 있었는데("NAVIGATING") 사람이 나타남!
                if self.mode == "NAVIGATING":
                    self.get_logger().warn(">>> 🚨 INTERRUPT: Person Detected! Cancelling RViz Goal!")
                    self.stop_robot() # 가던 길 멈춰!
                    self.mode = "FOLLOWING" # 이제부터 널 따라가겠어

                # 시나리오: 사람을 따라가는 중 ("FOLLOWING")
                if self.mode == "FOLLOWING":
                    if dist > 1.2: # 안전거리 밖이면 접근
                        target_pose = self.calculate_map_pose(x1, x2, y1, y2, dist - 0.8, msg.header)
                        if target_pose and self.should_update_goal(target_pose):
                            self.send_nav_goal(target_pose)
                            self.last_person_pose = target_pose
                            status_msg.data = f"Tracking Person: {dist:.1f}m"
                    else: # 너무 가까우면 정지
                        self.stop_robot()
                        status_msg.data = "Reached Person (Wait)"

                # 시각화
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) # 빨간 박스
                cv2.putText(frame, "TARGET", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        else:
            # 사람이 안 보일 때
            if self.mode == "FOLLOWING":
                # 사람 따라가다가 놓치면? -> 멈춤
                self.stop_robot()
                self.mode = "IDLE"
                status_msg.data = "Lost Person -> Stopped"
            elif self.mode == "NAVIGATING":
                # RViz 목표로 가는 중인데 사람이 없으면? -> 그냥 계속 감 (상관없음)
                status_msg.data = "Navigating to RViz Goal..."
            else:
                status_msg.data = "Idle (Waiting for Command)"

        self.pub_status.publish(status_msg)
        try: self.publisher_rviz.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))
        except: pass

    def should_update_goal(self, new_pose):
        if self.last_person_pose is None: return True
        dx = new_pose.pose.position.x - self.last_person_pose.pose.position.x
        dy = new_pose.pose.position.y - self.last_person_pose.pose.position.y
        return (dx*dx + dy*dy) > 0.04 # 20cm 이상 차이나면 업데이트

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
        point_cam.pose.position.x = x_cam
        point_cam.pose.position.y = y_cam
        point_cam.pose.position.z = z_cam
        point_cam.pose.orientation.w = 1.0

        try:
            transform = self.tf_buffer.lookup_transform('map', header_rgb.frame_id, rclpy.time.Time())
            target_pose = tf2_geometry_msgs.do_transform_pose(point_cam, transform)
            target_pose.pose.position.z = 0.0 
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
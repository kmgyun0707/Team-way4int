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
import traceback # 에러 위치 추적용

# 사람위치까지 이동가능 
class YoloCommander(Node):
    def __init__(self):
        super().__init__('yolo_commander_node')
        
        self.bridge = CvBridge()
        # 모델 경로가 맞는지 확인 필요 (같은 폴더에 yolov8n.pt가 있어야 함)
        self.model = YOLO('yolov8n.pt') 
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.current_goal_handle = None 

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 센서 데이터용 QoS
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        # RViz 명령 수신용 QoS
        goal_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 토픽 설정
        self.rgb_topic = '/front_stereo_camera/left/image_raw'
        self.depth_topic = '/front_stereo_camera/right/image_depth'
        
        self.create_subscription(Image, self.rgb_topic, self.rgb_callback, qos_profile=sensor_qos)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile=sensor_qos)
        self.create_subscription(PoseStamped, '/rviz_goal_pose', self.rviz_goal_callback, qos_profile=goal_qos)

        self.pub_status = self.create_publisher(String, '/yolo_status', 10)
        self.publisher_rviz = self.create_publisher(Image, 'person_detected_view', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_global_plan = self.create_publisher(Path, '/plan', 10)

        self.latest_depth = None
        self.mode = "IDLE"
        self.last_person_pose = None
        
        # 카메라 내부 파라미터 (Isaac Sim)
        self.fx = 640.0
        self.fy = 480.0
        self.cx = 640.0 / 2
        self.cy = 480.0 / 2

        self.get_logger().info(">>> YOLO Commander Ready: Waiting for '/rviz_goal_pose'...")

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except: pass

    # 1. 사용자가 RViz에서 목표를 찍으면 여기로 옴
    def rviz_goal_callback(self, msg):
        self.get_logger().info(">>> [USER CMD] Received Goal from RViz! Sending to Nav2...")
        self.mode = "NAVIGATING"
        self.is_moving_to_person = False
        self.send_nav_goal(msg)

    def send_nav_goal(self, pose_stamped):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_stamped
        
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(">>> [ERROR] Action Server 'navigate_to_pose' not ready! Is Nav2 running?")
            return

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(">>> [WARN] Goal Rejected by Nav2!")
            return
        self.get_logger().info(">>> [INFO] Goal Accepted! Robot Moving...")
        self.current_goal_handle = goal_handle

    def stop_robot(self):
        # 1. 속도 0 명령 전송
        stop_msg = Twist()
        self.pub_cmd_vel.publish(stop_msg)
        
        # 2. 실행 중인 Nav2 목표 취소
        if self.current_goal_handle and self.current_goal_handle.status == GoalStatus.STATUS_EXECUTING:
            self.current_goal_handle.cancel_goal_async()
        
        # 3. RViz 상의 경로 시각화 초기화
        empty_path = Path()
        empty_path.header.frame_id = 'map'
        empty_path.header.stamp = self.get_clock().now().to_msg()
        self.pub_global_plan.publish(empty_path)

    def rgb_callback(self, msg):
        status_msg = String()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
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
        
        if best_box:
            x1, y1, x2, y2 = best_box
            dist = self.get_distance_to_center(x1, y1, x2, y2)

            if dist > 0:
                if self.mode == "NAVIGATING":
                    self.get_logger().warn(f">>> 🚨 INTERRUPT: Person Detected ({dist:.1f}m)! Stopping RViz Goal.")
                    self.stop_robot()
                    self.mode = "FOLLOWING"
                    return 

                if self.mode == "FOLLOWING":
                    if dist > 0.5: #  수정1
                        target_pose = self.calculate_map_pose(x1, x2, y1, y2, dist - 1.0, msg.header)
                        
                        if target_pose:
                            if self.should_update_goal(target_pose):
                                self.send_nav_goal(target_pose)
                                self.last_person_pose = target_pose
                                status_msg.data = f"Tracking: {dist:.1f}m"
                        else:
                            status_msg.data = "TF Error (See Log)"
                    else:
                        self.stop_robot()
                        status_msg.data = "Reached Person"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "TARGET", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            if self.mode == "FOLLOWING":
                self.stop_robot()
                self.mode = "IDLE"
                status_msg.data = "Lost Person"
            elif self.mode == "NAVIGATING":
                status_msg.data = "Moving to RViz Goal..."
            else:
                status_msg.data = "IDLE"

        self.pub_status.publish(status_msg)
        try: self.publisher_rviz.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))
        except: pass

    def should_update_goal(self, new_pose):
        if self.last_person_pose is None: return True
        dx = new_pose.pose.position.x - self.last_person_pose.pose.position.x
        dy = new_pose.pose.position.y - self.last_person_pose.pose.position.y
        return (dx*dx + dy*dy) > 0.04

    def get_distance_to_center(self, x1, y1, x2, y2):
        if self.latest_depth is None: return 0.0
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        h, w = self.latest_depth.shape
        if not (0 <= cx < w and 0 <= cy < h): return 0.0
        roi = self.latest_depth[max(0, cy-5):min(h, cy+5), max(0, cx-5):min(w, cx+5)]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        return float(np.median(valid)) if len(valid) > 0 else 0.0

    # [최종 완성] 시간표 갱신이 포함된 calculate_map_pose
    def calculate_map_pose(self, x1, x2, y1, y2, dist, header_rgb):
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        z_cam = dist
        x_cam = (u - self.cx) * z_cam / self.fx
        y_cam = (v - self.cy) * z_cam / self.fy
        
        # 1. 원본 좌표 (시간 0으로 설정하여 TF 조회용)
        point_cam = PoseStamped()
        point_cam.header = header_rgb
        point_cam.header.frame_id = 'front_stereo_camera_left_rgb'
        point_cam.header.stamp = rclpy.time.Time(seconds=0).to_msg() # TF용 시간(과거/최신)

        point_cam.pose.position.x = x_cam
        point_cam.pose.position.y = y_cam
        point_cam.pose.position.z = z_cam
        point_cam.pose.orientation.w = 1.0

        try:
            # 2. 좌표 변환 (TF Buffer가 알아서 해줌)
            target_pose = self.tf_buffer.transform(
                point_cam, 
                'map', 
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # 3. 바닥으로 내리기
            target_pose.pose.position.z = 0.0 

            # [🚨 여기가 제일 중요!] Nav2에게 보내기 위해 "현재 시간"으로 갱신
            # 이걸 안 하면 Nav2가 명령을 무시합니다.
            target_pose.header.stamp = self.get_clock().now().to_msg()
            
            return target_pose

        except Exception as e:
            self.get_logger().warn(f"⚠️ Transform Failed: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = YoloCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
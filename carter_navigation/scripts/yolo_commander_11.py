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


# 시나리오2 사람인식 
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
        
        self.search_waypoints = [] 
        self.current_wp_index = 0  
        
        self.fx = 640.0
        self.fy = 480.0
        self.cx = 640.0 / 2
        self.cy = 480.0 / 2

        # [알고리즘 보정 변수]
        self.bag_detection_count = 0        
        self.bag_detection_threshold = 5   # 가방은 5번 연속 봐야 진짜로 인정
        self.person_max_detection_range = 3.0 # 3m 넘어가면 무시 (경로 꼬임 방지)

        self.get_logger().info(">>> YOLO Commander Ready (Lag-Robust Algorithm Applied)")

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
        self.search_waypoints = []
        self.current_wp_index = 0
        self.bag_detection_count = 0
        self.send_nav_goal(msg)

    def send_nav_goal(self, pose_stamped, callback_type="DEFAULT"):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_stamped
        
        # [수정] 렉 걸려서 서버 늦게 떠도 5초까지는 기다려줌
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(">>> [ERROR] Nav2 Action Server time out!")
            return
        
        future = self.nav_client.send_goal_async(goal_msg)
        
        if callback_type == "SEARCH":
            future.add_done_callback(self.goal_accepted_callback_search)
        else:
            future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            # self.get_logger().warn(">>> [WARN] Goal Rejected!")
            return
        self.current_goal_handle = goal_handle

    def goal_accepted_callback_search(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(">>> [WARN] Waypoint Rejected! Skipping...")
            self.process_next_waypoint()
            return
        
        self.current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback_search)

    def goal_result_callback_search(self, future):
        if self.mode != "SEARCHING":
            return

        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f">>> [SEARCH] Waypoint {self.current_wp_index + 1} Reached!")
        
        self.process_next_waypoint()

    def process_next_waypoint(self):
        if self.mode != "SEARCHING": return 

        self.current_wp_index += 1
        if self.current_wp_index < len(self.search_waypoints):
            next_pose = self.search_waypoints[self.current_wp_index]
            next_pose.header.stamp = self.get_clock().now().to_msg()
            self.send_nav_goal(next_pose, callback_type="SEARCH")
        else:
            self.get_logger().info(">>> [SEARCH] All waypoints searched. Return to IDLE.")
            self.mode = "IDLE"
            self.stop_robot()

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

    def generate_search_waypoints(self, center_pose):
        waypoints = []
        radius = 1.0 
        angles = [math.pi/2, 0, -math.pi/2, math.pi] 
        
        cx = center_pose.pose.position.x
        cy = center_pose.pose.position.y

        for angle in angles:
            wp = PoseStamped()
            wp.header.frame_id = 'map'
            
            wx = cx + radius * math.cos(angle)
            wy = cy + radius * math.sin(angle)
            wp.pose.position.x = wx
            wp.pose.position.y = wy
            wp.pose.position.z = 0.0

            yaw = math.atan2(cy - wy, cx - wx)
            wp.pose.orientation.z = math.sin(yaw / 2.0)
            wp.pose.orientation.w = math.cos(yaw / 2.0)
            
            waypoints.append(wp)
            
        return waypoints

    def get_distance_to_center(self, x1, y1, x2, y2):
        if self.latest_depth is None: return 0.0
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        h, w = self.latest_depth.shape
        if not (0 <= cx < w and 0 <= cy < h): return 0.0
        roi = self.latest_depth[max(0, cy-5):min(h, cy+5), max(0, cx-5):min(w, cx+5)]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        return float(np.median(valid)) if len(valid) > 0 else 0.0

    # [중요] 지연과 오차를 보정하는 강력한 좌표 계산 함수
    def calculate_map_pose(self, x1, x2, y1, y2, dist, header_rgb):
        stop_offset = 0.3  # 사람 0.3m 앞에서 정지
        
        if dist <= stop_offset:
            return None

        # 1차 시도: 정석적인 TF 변환
        try:
            z_target = dist - stop_offset
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            
            x_cam = (u - self.cx) * z_target / self.fx
            y_cam = (v - self.cy) * z_target / self.fy
            z_cam = z_target

            point_cam = PoseStamped()
            point_cam.header = header_rgb
            
            # 프레임 이름의 '/' 문제 해결
            if point_cam.header.frame_id.startswith('/'):
                point_cam.header.frame_id = point_cam.header.frame_id[1:]
                
            point_cam.header.stamp = rclpy.time.Time(seconds=0).to_msg()
            point_cam.pose.position.x = x_cam
            point_cam.pose.position.y = y_cam
            point_cam.pose.position.z = z_cam
            point_cam.pose.orientation.w = 1.0
            
            # 타임아웃을 1.0초로 넉넉하게
            target_pose = self.tf_buffer.transform(
                point_cam, 
                'map', 
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            target_pose.pose.position.z = 0.0 
            return target_pose

        except Exception as e:
            # 2차 시도: TF 실패 시 비상 대책 (수동 계산)
            # 지연이 심하면 이 코드가 실행될 확률이 높음
            try:
                u_center = (x1 + x2) / 2
                u_offset = self.cx - u_center
                
                y_offset = 1 * (u_offset * dist / self.fx) #반전 적용했다가 -1을 1로 수정함
                
                fallback_pose = PoseStamped()
                # 로봇 몸통(base_link) 기준으로 계산
                fallback_pose.header.frame_id = 'base_link'
                fallback_pose.header.stamp = self.get_clock().now().to_msg()
                
                fallback_pose.pose.position.x = dist - stop_offset # 앞으로
                fallback_pose.pose.position.y = y_offset           # 옆으로 (반전 적용됨)
                fallback_pose.pose.position.z = 0.0
                fallback_pose.pose.orientation.w = 1.0
                
                target_pose = self.tf_buffer.transform(
                    fallback_pose, 
                    'map', 
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                target_pose.pose.position.z = 0.0
                return target_pose
                
            except Exception as e2:
                # 이것마저 실패하면 로그 찍고 무시
                self.get_logger().error(f"Fallback Calc Failed: {e2}")
                return None

    def rgb_callback(self, msg):
        status_msg = String()
        try: frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except: return

        # [수정] 인식 민감도 0.4
        results = self.model(frame, classes=[0, 24, 26, 28], verbose=False, conf=0.4)
        
        best_box = None
        person_box = None
        bag_box = None
        detected_cls = -1
        
        if any(len(r.boxes) > 0 for r in results):
            for r in results:
                for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                    cls_id = int(cls)
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2-x1) * (y2-y1)

                    if cls_id == 0: 
                        if person_box is None or area > person_box['area']:
                            person_box = {'box': (x1, y1, x2, y2), 'area': area, 'cls': 0}
                    elif cls_id in [24, 26, 28]: 
                        if bag_box is None or area > bag_box['area']:
                            bag_box = {'box': (x1, y1, x2, y2), 'area': area, 'cls': cls_id}

        if person_box:
            best_box = person_box['box']
            detected_cls = 0
            self.bag_detection_count = 0 
        elif bag_box:
            best_box = bag_box['box']
            detected_cls = bag_box['cls']
            self.bag_detection_count += 1
        else:
            self.bag_detection_count = 0

        # 화면 그리기
        dist = 0.0
        if best_box:
            x1, y1, x2, y2 = best_box
            dist = self.get_distance_to_center(x1, y1, x2, y2)
            label = self.model.names[detected_cls]
            color = (0, 0, 255) if detected_cls == 0 else (255, 0, 0)
            
            debug_text = f"{label} {dist:.1f}m"
            if detected_cls in [24, 26, 28]:
                debug_text += f" ({self.bag_detection_count}/{self.bag_detection_threshold})"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, debug_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        try: 
            ros_img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            ros_img_msg.header = msg.header
            self.pub_yolo_img.publish(ros_img_msg)
        except: pass

        # ------------------------------------------------------------------
        # [핵심 행동 로직]
        # ------------------------------------------------------------------
        if best_box and dist > 0:
            
            # [CASE 1] 사람 발견
            if detected_cls == 0:
                # 1. IDLE(RViz 명령 전)일 때는 무시 -> 시작하자마자 튀어나가는 것 방지
                if self.mode != "IDLE":
                    # 2. 3m 이내일 때만 반응 -> 멀리 있는 귀신 쫓아가는 것 방지
                    if dist <= self.person_max_detection_range:
                        if dist > 0.6: # 0.6m보다 멀면 접근
                            self.get_logger().info(f">>> 🙋 Approach: {dist:.1f}m left.")
                            target_pose = self.calculate_map_pose(x1, x2, y1, y2, dist, msg.header)
                            
                            if target_pose:
                                self.mode = "APPROACHING_PERSON"
                                self.send_nav_goal(target_pose, callback_type="PERSON")
                        else:
                            # 충분히 가까워지면 정지
                            if self.mode == "APPROACHING_PERSON" and dist <= 0.6:
                                self.get_logger().info(">>> Reached Person! Stopping.")
                                self.stop_robot()
                                self.mode = "IDLE"

            # [CASE 2] 가방 발견
            elif detected_cls in [24, 26, 28]:
                # 주행 중일 때만 반응
                if self.mode == "NAVIGATING":
                    # 3. 가방은 5번 이상 연속으로 봐야 진짜로 인정 -> 노이즈 방지
                    if self.bag_detection_count >= self.bag_detection_threshold:
                        self.get_logger().warn(f">>> 🎒 Bag Confirmed ({dist:.1f}m)! Search.")
                        self.stop_robot()
                        
                        bag_pose = self.calculate_map_pose(x1, x2, y1, y2, dist, msg.header)
                        if bag_pose:
                            self.search_waypoints = self.generate_search_waypoints(bag_pose)
                            self.current_wp_index = 0
                            self.mode = "SEARCHING"
                            
                            start_pose = self.search_waypoints[0]
                            start_pose.header.stamp = self.get_clock().now().to_msg()
                            self.send_nav_goal(start_pose, callback_type="SEARCH")
                            
                            self.bag_detection_count = 0
                        else:
                            self.mode = "NAVIGATING"

        if self.mode == "APPROACHING_PERSON":
            status_msg.data = f">>> Approaching! {dist:.1f}m <<<"
        elif self.mode == "SEARCHING":
            status_msg.data = f"Searching Pattern: {self.current_wp_index + 1}/4"
        elif self.mode == "NAVIGATING":
            status_msg.data = "Moving to Goal..."
        else:
            status_msg.data = "IDLE (Waiting for Goal)"

        self.pub_status.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, OccupancyGrid
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

#시나리오 2: 사람과 가방 모두 인식하여 동작하는 코드
#1m이내로 사람이 보이면 즉시 접근
#가방이 보이면 그 주변을 1m 반경으로 8군데 웨이포인트를 생성하여 탐색
#유효성 검사: Costmap을 구독하여 벽(장애물)인 지점은 제외
#최단거리 정렬(Greedy): 현재 로봇 위치에서 가까운 순서로 웨이포인트 정렬

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
        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.rgb_topic = '/front_stereo_camera/left/image_raw'
        self.depth_topic = '/front_stereo_camera/right/image_depth'
        
        self.create_subscription(Image, self.rgb_topic, self.rgb_callback, qos_profile=sensor_qos)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile=sensor_qos)
        self.create_subscription(PoseStamped, '/rviz_goal_pose', self.rviz_goal_callback, qos_profile=goal_qos)
        
        # [추가] 장애물 확인을 위한 Costmap 구독 (Global Costmap)
        self.costmap = None
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.costmap_callback, qos_profile=map_qos)

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

        self.get_logger().info(">>> YOLO Smart Commander Ready (Obstacle-Aware Search)")

    def costmap_callback(self, msg):
        # 맵 데이터를 저장해둡니다.
        self.costmap = msg

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
        self.send_nav_goal(msg)

    def send_nav_goal(self, pose_stamped, callback_type="DEFAULT"):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_stamped
        
        if not self.nav_client.wait_for_server(timeout_sec=0.5): return
        
        self.get_logger().info(f">>> [MOVE] Sending Goal ({callback_type})...")
        future = self.nav_client.send_goal_async(goal_msg)
        
        if callback_type == "SEARCH":
            future.add_done_callback(self.goal_accepted_callback_search)
        elif callback_type == "PERSON":
            future.add_done_callback(self.goal_response_callback)
        else:
            future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(">>> [WARN] Goal Rejected!")
            return
        self.current_goal_handle = goal_handle

    def goal_accepted_callback_search(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(">>> [WARN] Waypoint Rejected (Maybe Wall?)! Skipping to next...")
            self.process_next_waypoint() # 거절되면 바로 다음 최적점으로
            return
        
        self.current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback_search)

    def goal_result_callback_search(self, future):
        if self.mode != "SEARCHING": return

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
            self.get_logger().info(">>> [SEARCH] All valid waypoints searched. IDLE.")
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

    # =========================================================================
    # [핵심] 스마트 탐색 알고리즘
    # 1. 8방향(45도) 후보 생성
    # 2. Costmap 확인하여 벽(장애물) 제거
    # 3. 현재 로봇 위치에서 가까운 순서로 정렬 (Nearest Neighbor)
    # =========================================================================
    def generate_smart_waypoints(self, center_pose):
        candidates = []
        radius = 1.0 
        # 8방향 (0도부터 360도까지 45도 간격)
        angles = np.arange(0, 2*math.pi, math.pi/4) 
        
        cx = center_pose.pose.position.x
        cy = center_pose.pose.position.y

        # 1. 후보 좌표 생성 및 맵 유효성 검사
        valid_waypoints = []
        
        for angle in angles:
            wx = cx + radius * math.cos(angle)
            wy = cy + radius * math.sin(angle)
            
            # 맵 상에서 장애물인지 확인
            if self.is_point_safe(wx, wy):
                wp = PoseStamped()
                wp.header.frame_id = 'map'
                wp.pose.position.x = wx
                wp.pose.position.y = wy
                wp.pose.position.z = 0.0

                # 로봇이 가방을 바라보도록
                yaw = math.atan2(cy - wy, cx - wx)
                wp.pose.orientation.z = math.sin(yaw / 2.0)
                wp.pose.orientation.w = math.cos(yaw / 2.0)
                
                valid_waypoints.append(wp)
            else:
                # 디버깅용 로그 (선택사항)
                pass # 벽이라서 제외됨

        if not valid_waypoints:
            self.get_logger().warn(">>> [WARN] No valid search points found around bag!")
            return []

        # 2. 거리 기반 정렬 (Nearest Neighbor)
        # 현재 로봇 위치를 가져옴
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time(seconds=0))
            rx = transform.transform.translation.x
            ry = transform.transform.translation.y
            
            # 로봇과 각 웨이포인트 사이의 거리 계산 후 정렬 (가까운 순)
            valid_waypoints.sort(key=lambda wp: math.sqrt((wp.pose.position.x - rx)**2 + (wp.pose.position.y - ry)**2))
            
            self.get_logger().info(f">>> [SMART SEARCH] Generated {len(valid_waypoints)} valid points (sorted by distance).")
            return valid_waypoints
            
        except:
            self.get_logger().warn("Failed to get robot pose for sorting. Using default order.")
            return valid_waypoints

    def is_point_safe(self, x, y):
        """ Costmap 데이터를 이용해 (x,y)가 갈 수 있는 곳인지 확인 """
        if self.costmap is None: 
            return True # 맵 데이터 없으면 일단 간다고 가정
        
        # World 좌표 -> Grid 인덱스 변환
        resolution = self.costmap.info.resolution
        origin_x = self.costmap.info.origin.position.x
        origin_y = self.costmap.info.origin.position.y
        width = self.costmap.info.width
        height = self.costmap.info.height

        col = int((x - origin_x) / resolution)
        row = int((y - origin_y) / resolution)

        if col < 0 or col >= width or row < 0 or row >= height:
            return False # 맵 밖임

        # Cost 가져오기 (0~100, -1은 Unknown)
        index = row * width + col
        cost = self.costmap.data[index]

        # 일반적으로 Costmap에서:
        # 0: Free space
        # 100: Lethal obstacle
        # > 50: 위험 구역 (Inflation radius 등)
        # 여기서는 안전하게 50 이상이면 가지 않도록 설정
        if cost > 50 or cost == -1: 
            return False
        
        return True

    def rgb_callback(self, msg):
        status_msg = String()
        try: frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except: return

        results = self.model(frame, classes=[0, 24, 26, 28], verbose=False, conf=0.6)
        
        best_box = None
        max_area = 0
        detected_cls = -1
        person_box = None
        bag_box = None
        
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
        elif bag_box:
            best_box = bag_box['box']
            detected_cls = bag_box['cls']

        dist = 0.0
        if best_box:
            x1, y1, x2, y2 = best_box
            dist = self.get_distance_to_center(x1, y1, x2, y2)
            label = self.model.names[detected_cls]
            color = (0, 0, 255) if detected_cls == 0 else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{label} {dist:.1f}m", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        try: 
            ros_img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            ros_img_msg.header = msg.header
            self.pub_yolo_img.publish(ros_img_msg)
        except: pass

        if best_box and dist > 0:
            if detected_cls == 0:
                if self.mode != "APPROACHING_PERSON":
                    self.get_logger().info(f">>> 🙋 Person Detected ({dist:.1f}m)! Stop & Approach.")
                    self.stop_robot()
                    target_pose = self.calculate_map_pose(x1, x2, y1, y2, dist, msg.header)
                    if target_pose:
                        self.mode = "APPROACHING_PERSON"
                        self.send_nav_goal(target_pose, callback_type="PERSON")

            elif detected_cls in [24, 26, 28]:
                if self.mode == "NAVIGATING":
                    self.get_logger().warn(f">>> 🎒 Bag Detected ({dist:.1f}m)! Starting Smart Search.")
                    self.stop_robot()
                    
                    bag_pose = self.calculate_map_pose(x1, x2, y1, y2, dist, msg.header)
                    if bag_pose:
                        # [변경] 기존 단순 생성 -> 스마트 생성 함수 호출
                        self.search_waypoints = self.generate_smart_waypoints(bag_pose)
                        
                        if len(self.search_waypoints) > 0:
                            self.current_wp_index = 0
                            self.mode = "SEARCHING"
                            start_pose = self.search_waypoints[0]
                            start_pose.header.stamp = self.get_clock().now().to_msg()
                            self.send_nav_goal(start_pose, callback_type="SEARCH")
                        else:
                            self.get_logger().warn(">>> No valid path found. Continuing navigation.")
                            self.mode = "NAVIGATING"

        if self.mode == "APPROACHING_PERSON":
            status_msg.data = ">>> Approaching Person! <<<"
        elif self.mode == "SEARCHING":
            status_msg.data = f"Smart Search: {self.current_wp_index + 1}/{len(self.search_waypoints)}"
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
            target_pose = self.tf_buffer.transform(point_cam, 'map', timeout=rclpy.duration.Duration(seconds=0.1))
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
#!/usr/bin/env python3

#욜로 수정(확인용)
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose  # [변경] Nav2 액션 직접 사용
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class YoloCommander(Node):
    def __init__(self):
        super().__init__('yolo_commander_node')
        
        # 1. 초기 설정
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt') 
        
        # [핵심 변경] BasicNavigator 제거 -> ActionClient 사용
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.current_goal_handle = None # 현재 이동 중인 목표를 기억하는 변수

        # QoS 설정
        qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 2. 토픽 구독
        self.target_topic = '/front_stereo_camera/left/image_raw' 
        self.create_subscription(Image, self.target_topic, self.rgb_callback, qos_profile=qos_policy)
        
        # RViz 목표 가로채기 (/goal_pose)
        self.create_subscription(PoseStamped, '/goal_pose', self.rviz_goal_callback, 10)

        # 3. 디버깅용 발행기
        self.pub_status = self.create_publisher(String, '/yolo_status', 10)
        self.publisher_rviz = self.create_publisher(Image, 'person_detected_view', 10)

        # 생존 신고 타이머
        self.last_img_time = 0
        self.create_timer(1.0, self.timer_callback)

        self.get_logger().info(">>> YOLO Commander (ActionClient Ver) Ready!")

    def timer_callback(self):
        # 카메라 연결 상태 확인
        time_diff = self.get_clock().now().nanoseconds - self.last_img_time
        if self.last_img_time > 0 and (time_diff / 1e9) > 3.0:
            msg = String()
            msg.data = "WARNING: Camera Disconnected?"
            self.pub_status.publish(msg)

    # [기능 1] RViz 목표를 받아서 -> Nav2에게 정식 요청 보내기
    def rviz_goal_callback(self, msg):
        self.get_logger().info("New Goal Received from RViz! Sending to Nav2...")
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = msg
        
        # Nav2 액션 서버 대기
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Nav2 Action Server not ready!")
            return

        # 비동기(Async)로 목표 전송
        self.future = self.nav_client.send_goal_async(goal_msg)
        self.future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected by Nav2 :(')
            return

        self.get_logger().info('Goal Accepted by Nav2! Robot should move.')
        self.current_goal_handle = goal_handle # [중요] 핸들을 저장해야 나중에 취소 가능

    # [기능 2] YOLO 감지 및 정지 로직
    def rgb_callback(self, msg):
        self.last_img_time = self.get_clock().now().nanoseconds
        status_msg = String()
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except: return

        results = self.model(frame, classes=[0], verbose=False, conf=0.5)
        person_detected = False

        for result in results:
            if len(result.boxes) > 0:
                person_detected = True
                # 시각화
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "EMERGENCY STOP", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # ====================================================
                # [수정된 정지 로직] 저장된 Goal Handle이 있으면 취소 요청
                # ====================================================
                if self.current_goal_handle is not None:
                    # 이미 취소된 상태가 아니라면
                    if self.current_goal_handle.status == GoalStatus.STATUS_EXECUTING or \
                       self.current_goal_handle.status == GoalStatus.STATUS_ACCEPTED:
                        
                        self.get_logger().warn("!!! PERSON FOUND -> CANCELLING GOAL !!!")
                        future = self.current_goal_handle.cancel_goal_async()
                        future.add_done_callback(self.cancel_done_callback)
                        
                        self.current_goal_handle = None # 핸들 초기화 (중복 취소 방지)
                        status_msg.data = "ACTION: STOP TRIGGERED (Cancelling...)"
                    else:
                        status_msg.data = "DETECTED: Person found (Already Stopping/Idle)"
                else:
                    status_msg.data = "DETECTED: Person found (No Active Goal)"
            else:
                status_msg.data = "OK: Scanning... (Moving)" if self.current_goal_handle else "OK: Scanning... (Idle)"

        self.pub_status.publish(status_msg)
        
        try:
            self.publisher_rviz.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))
        except: pass

    def cancel_done_callback(self, future):
        self.get_logger().info("Goal successfully cancelled!")

def main(args=None):
    rclpy.init(args=args)
    node = YoloCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
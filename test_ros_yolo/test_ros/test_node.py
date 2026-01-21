import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ultralytics import YOLO

class IsaacSimYoloNode(Node):
    def __init__(self):
        super().__init__('isaac_sim_yolo_node')
        self.bridge = CvBridge()
        
        # 16GB VRAM의 위력을 쓰기 위해 Large 모델 로드
        self.model = YOLO('yolov8n.pt') 
        self.model.to('cuda') # RTX 5060 강제 할당

        # 지연 방지를 위한 고성능 QoS 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, # 속도 우선
            history=HistoryPolicy.KEEP_LAST,
            depth=1 # 최신 프레임 1개만 유지
        )

        # 아이작 심의 Topic Name과 반드시 일치해야 함
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw', 
            self.listener_callback,
            qos_profile
        )
        self.get_logger().info('Isaac Sim YOLO Node with RTX 5060 Started.')

    def listener_callback(self, msg):
        # 1. 이미지 변환
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 2. YOLO 추론 (imgsz는 Render Product의 해상도와 맞추는 것이 좋습니다)
        results = self.model(cv_image, verbose=False, imgsz=640)
        
        # 3. 지연 시간 측정 (시뮬레이션 시간과 동기화 확인)
        latency = (self.get_clock().now() - rclpy.time.Time.from_msg(msg.header.stamp)).nanoseconds / 1e6
        self.get_logger().info(f'Latency: {latency:.2f}ms | Detected: {len(results[0].boxes)}')

def main(args=None):
    rclpy.init(args=args)
    node = IsaacSimYoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
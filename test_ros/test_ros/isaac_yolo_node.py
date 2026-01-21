import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ultralytics import YOLO
import torch

class IsaacYoloNode(Node):
    def __init__(self):
        super().__init__('isaac_yolo_node')
        self.bridge = CvBridge()
        
        # 16GB VRAM 활용: 정밀도가 높은 Large 모델 권장
        self.model = YOLO('yolov8l.pt') 
        if torch.cuda.is_available():
            self.model.to('cuda')
            self.get_logger().info("Using GPU: RTX 5060")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 아이작 심의 토픽 구독
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, qos)

        # 표준 인식 결과 발행 (로봇 제어팀이 사용할 데이터)
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/detections', qos)

    def image_callback(self, msg):
        # 1. 이미지 변환 및 전처리
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 2. 고해상도 추론
        results = self.model(cv_image, verbose=False, imgsz=1280)
        
        # 3. 결과 메시지 구성
        detection_array = Detection2DArray()
        detection_array.header = msg.header # 타임스탬프 동기화

        for result in results[0].boxes:
            det = Detection2D()
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(int(result.cls))
            hyp.hypothesis.score = float(result.conf)
            det.results.append(hyp)

            # Center X, Y, Size X, Y
            box = result.xywh[0]
            det.bbox.center.position.x = float(box[0])
            det.bbox.center.position.y = float(box[1])
            det.bbox.size_x = float(box[2])
            det.bbox.size_y = float(box[3])
            detection_array.detections.append(det)

        self.detection_pub.publish(detection_array)

        # 지연 시간 로그 (16GB VRAM의 위력 확인)
        latency = (self.get_clock().now() - rclpy.time.Time.from_msg(msg.header.stamp)).nanoseconds / 1e6
        self.get_logger().info(f'End-to-End Latency: {latency:.2f}ms')

def main(args=None):
    rclpy.init(args=args)
    node = IsaacYoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
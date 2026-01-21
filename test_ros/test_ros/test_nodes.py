import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class WebcamTestNode(Node):
    def __init__(self):
        super().__init__('webcam_test_node')
        self.bridge = CvBridge()
        
        # 1. 백엔드를 V4L2로 강제 지정하여 GStreamer 오류 방지
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) # YOLO 표준 480 혹은 640
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 2. 정의한 QoS 적용
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.publisher = self.create_publisher(Image, 'camera/image_raw', qos_profile)
        self.subscription = self.create_subscription(
            Image, 'camera/image_raw', self.listener_callback, qos_profile)
        
        self.timer = self.create_timer(0.033, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_frame"
            self.publisher.publish(msg)
        else:
            self.get_logger().error('웹캠 데이터를 읽을 수 없습니다.')

    def listener_callback(self, msg):
        receive_time = self.get_clock().now()
        send_time = rclpy.time.Time.from_msg(msg.header.stamp)
        latency = (receive_time - send_time).nanoseconds / 1e6
        self.get_logger().info(f'Latency: {latency:.2f} ms')

def main(args=None):
    rclpy.init(args=args)
    node = WebcamTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.cap.isOpened():
            node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
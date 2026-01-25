#!/usr/bin/env python3


#초기 버전
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
from geometry_msgs.msg import PointStamped


class Processing(Node):
    def __init__(self):
        super().__init__('proccese_node')
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt') 

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.width = 640
        self.height = 480 
        self.fx = 640.0 # Focal Length (예시)
        self.fy = 480.0
        self.cx = self.width / 2
        self.cy = self.height / 2

        self.latest_depth = None

        #ros2 subscription 
        self.create_subscription(Image, '/depth', self.depth_callback, 10)
        self.create_subscription(Image, '/rgb', self.rgb_callback, 10)
        self.get_logger().info("토픽 분석 시작")

        #ros2 publisher 
        self.publisher_rviz = self.create_publisher(Image, 'person_depth',10)
        self.publisher_depth = self.create_publisher(PointStamped, 'depth_to_car',10)


    def depth_callback(self, msg):
        #뎁스 데이터 수신 
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
    
    def rgb_callback(self,msg):
        if self.latest_depth is None: # [추가] 데이터 수신 전이면 리턴
            return
       
       
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        results = self.model(frame, classes=[0], verbose=False, conf=0.3)


        
        for result in results:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                
                # 검출 영역 거리 계산 
                roi = self.latest_depth[y1:y2, x1:x2]
                valid_vals = roi[np.isfinite(roi) & (roi > 0)]
                #바운딩 박스 생성 
                if len(valid_vals) > 0:
                    
                    dist = np.median(valid_vals)
                    label = f"Person: {dist:.2f}m"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


                    target_msg = PointStamped()
                    target_msg.header.stamp = self.get_clock().now().to_msg()
                    target_msg.header.frame_id = "sim_camera"
                    u_center = (x1 + x2) / 2
                    v_center = (y1 + y2) / 2

                    target_msg.point.z = float(dist)
                    target_msg.point.x = (u_center - self.cx) * target_msg.point.z / self.fx
                    target_msg.point.y = (v_center - self.cy) * target_msg.point.z / self.fy

                    self.publisher_depth.publish(target_msg)
                    
        
        cv2.imshow("Isaac Sim Total Analysis (Distance & Depth)", frame)
        cv2.waitKey(1)
        
        yolo_result = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        yolo_result.header = msg.header
        self.publisher_rviz.publish(yolo_result)


def main(args=None):
    rclpy.init(args=args)
    node = Processing()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

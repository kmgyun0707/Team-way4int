#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf2_ros
import numpy as np
import cv2

class DebugNode(Node):
    def __init__(self):
        super().__init__('debug_node')
        self.bridge = CvBridge()

        # 1. 토픽 이름 설정 (사용자 설정에 맞춤)
        self.rgb_topic = '/front_stereo_camera/left/image_raw'
        self.depth_topic = '/front_stereo_camera/right/image_depth'
        
        # 2. TF 리스너 (위치 관계 확인용)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 3. 구독 시작
        self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)

        self.rgb_received = False
        self.depth_received = False
        
        print("\n" + "="*50)
        print("🔍 진단 시작... (데이터 기다리는 중)")
        print(f"Target RGB: {self.rgb_topic}")
        print(f"Target Depth: {self.depth_topic}")
        print("="*50 + "\n")

    def rgb_callback(self, msg):
        if not self.rgb_received:
            print(f"✅ [RGB] 연결 성공! (Frame ID: {msg.header.frame_id})")
            self.rgb_received = True
            
            # TF 확인 (RGB 프레임 <-> Map)
            try:
                # 0.1초 안에 찾을 수 있는지 테스트
                transform = self.tf_buffer.lookup_transform('map', msg.header.frame_id, rclpy.time.Time())
                print(f"✅ [TF] 좌표 변환 가능! ('map' -> '{msg.header.frame_id}')")
            except Exception as e:
                print(f"❌ [TF] 좌표 변환 실패! (원인: {e})")
                print("   -> 힌트: use_sim_time:=True 를 썼나요? 아니면 static_transform이 꺼졌나요?")

    def depth_callback(self, msg):
        try:
            # Depth 데이터 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # 중앙값 추출
            h, w = cv_image.shape
            center_val = cv_image[int(h/2), int(w/2)]
            max_val = np.max(cv_image)
            min_val = np.min(cv_image)

            if not self.depth_received:
                print(f"✅ [Depth] 연결 성공! (Frame ID: {msg.header.frame_id})")
                print(f"   -> 데이터 통계: Min={min_val:.2f}m, Max={max_val:.2f}m, Center={center_val:.2f}m")
                
                if max_val == 0:
                    print("⚠️ [경고] Depth 값이 전부 0입니다! (카메라가 렌더링을 못하고 있음)")
                elif msg.header.frame_id != "front_stereo_camera_left_optical":
                    print(f"⚠️ [경고] Frame ID가 '{msg.header.frame_id}' 입니다.")
                    print("   -> 우리가 원한 건 'front_stereo_camera_left_optical' 입니다.")
                    print("   -> Action Graph에서 Constant String 연결을 다시 확인하세요.")
                else:
                    print("👍 [Depth] 상태 완벽함!")
                
                self.depth_received = True

        except Exception as e:
            print(f"❌ [Depth] 에러 발생: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DebugNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

# Import jetseg - Thư viện Human Segmentation tối ưu cho Jetson
try:
    from jetseg import HumanSeg
except ImportError:
    raise ImportError("❌ Không tìm thấy thư viện 'jetseg'. Vui lòng cài đặt: pip install jetseg-*.whl")

class HumanSegmentNode(Node):
    def __init__(self):
        super().__init__('human_segment_node')
        
        # Khai báo tham số
        self.declare_parameter('input_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('output_mask_topic', '/b_cam_ai/human_mask')
        self.declare_parameter('output_viz_topic', '/b_cam_ai/human_viz')
        self.declare_parameter('use_fp16', True)
        self.declare_parameter('threshold', 0.5)

        # Lấy giá trị tham số
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_mask_topic = self.get_parameter('output_mask_topic').get_parameter_value().string_value
        output_viz_topic = self.get_parameter('output_viz_topic').get_parameter_value().string_value
        use_fp16 = self.get_parameter('use_fp16').get_parameter_value().bool_value
        self.threshold = self.get_parameter('threshold').get_parameter_value().double_value

        # Khởi tạo CV Bridge
        self.bridge = CvBridge()

        # Khởi tạo JetSeg Engine
        self.get_logger().info(f"⏳ Đang khởi tạo JetSeg (FP16={use_fp16})...")
        try:
            # JetSeg tự động tìm model trong package của nó
            # Lần đầu chạy sẽ mất 1-2 phút để build engine TensorRT
            self.seg_engine = HumanSeg(use_fp16=use_fp16)
            self.get_logger().info("✅ JetSeg Engine đã sẵn sàng!")
        except Exception as e:
            self.get_logger().error(f"❌ Lỗi khởi tạo JetSeg: {e}")
            raise e

        # Subscribers & Publishers
        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self.listener_callback,
            10)
        
        self.mask_publisher = self.create_publisher(Image, output_mask_topic, 10)
        self.viz_publisher = self.create_publisher(Image, output_viz_topic, 10)
        
        self.get_logger().info(f"🚀 Đang lắng nghe: {input_topic}")

    def listener_callback(self, msg):
        start_time = time.time()
        
        try:
            # 1. Chuyển đổi ROS Image -> OpenCV Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Lỗi convert ảnh: {e}")
            return

        # 2. Chạy Inference (JetSeg)
        # Trả về mask 0 hoặc 255
        mask = self.seg_engine.predict(cv_image, threshold=self.threshold)

        if mask is None:
            self.get_logger().warn("JetSeg trả về None mask")
            return

        # 3. Publish Mask (Mono8)
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(mask, "mono8")
            mask_msg.header = msg.header # Giữ nguyên timestamp
            self.mask_publisher.publish(mask_msg)
        except Exception as e:
            self.get_logger().error(f"Lỗi publish mask: {e}")

        # 4. (Optional) Tạo ảnh Visualization (Nền xanh) và Publish
        # Chỉ xử lý nếu có người đang subscribe topic visualization để tiết kiệm CPU
        if self.viz_publisher.get_subscription_count() > 0:
            # Dùng hàm tiện ích của JetSeg để tách nền
            viz_img = self.seg_engine.remove_background(cv_image, mask, bg_color=(0, 255, 0))
            
            # Vẽ FPS lên ảnh
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(viz_img, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            try:
                viz_msg = self.bridge.cv2_to_imgmsg(viz_img, "bgr8")
                viz_msg.header = msg.header
                self.viz_publisher.publish(viz_msg)
            except Exception as e:
                self.get_logger().error(f"Lỗi publish viz: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = HumanSegmentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
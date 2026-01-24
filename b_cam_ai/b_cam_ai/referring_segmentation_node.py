import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory

import torch
import cv2
import numpy as np
import os
import random

# Import modules từ package
from b_cam_ai.models import build_model
from b_cam_ai.datasets import SimpleTokenizer, VNPromptGenerator

class ReferringSegmentationNode(Node):
    def __init__(self):
        super().__init__('referring_segmentation_node')

        # --- TÌM PATH ---
        try:
            pkg_share = get_package_share_directory('b_cam_ai')
            default_model_path = os.path.join(pkg_share, 'weights', 'best_model.pth')
        except Exception:
            default_model_path = './weights/best_model.pth'

        # --- PARAMETERS ---
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('img_size', 320)
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.declare_parameter('input_topic', '/camera/camera/color/image_raw')
        
        # Tham số quan trọng: Danh sách vật thể cần tìm (ngăn cách bởi dấu phẩy)
        self.declare_parameter('target_objects', 'người') 

        # Lấy giá trị
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.img_size = self.get_parameter('img_size').get_parameter_value().integer_value
        self.threshold = self.get_parameter('threshold').get_parameter_value().double_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        target_str = self.get_parameter('target_objects').get_parameter_value().string_value

        self.get_logger().info(f"Target Objects: {target_str}")
        self.get_logger().info(f"Device: {self.device}")

        # --- INIT PROMPT GENERATOR ---
        # Tự động tìm file template trong thư mục cài đặt của b_cam_ai/data
        # VNPromptGenerator trong datasets.py đã xử lý logic tìm file tương đối
        try:
            self.prompt_gen = VNPromptGenerator()
            self.get_logger().info("Prompt Generator: OK")
        except Exception as e:
            self.get_logger().warn(f"Lỗi khởi tạo Prompt Generator: {e}. Dùng template mặc định.")
            self.prompt_gen = None

        # --- SINH PROMPT TỰ ĐỘNG ---
        self.current_prompt = self.generate_auto_prompt(target_str)
        self.get_logger().info(f"AI Prompt -> '{self.current_prompt}'")

        # --- INIT AI MODEL ---
        self.tokenizer = SimpleTokenizer(max_length=20)
        self.model = build_model(self.tokenizer.vocab_size, device=self.device)
        
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=False)
            self.get_logger().info("Đã load weights.")
        else:
            self.get_logger().error(f"Không tìm thấy weights: {self.model_path}")

        # --- FIX LỖI BATCHNORM ---
        self.model.eval() 
        # -------------------------

        # --- ROS UTILS ---
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, input_topic, self.image_callback, qos_profile_sensor_data
        )
        self.pub_mask = self.create_publisher(Image, 'b_cam_ai/ref_seg_mask', 10)
        self.pub_overlay = self.create_publisher(Image, 'b_cam_ai/ref_seg_overlay', 10)
        
        # Precalc setup
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.mask_color = np.array([0, 255, 0], dtype=np.uint8)

    def generate_auto_prompt(self, target_str):
        """
        Input: "người, cái cốc"
        Output: "hãy tìm người và cái cốc" (Random template)
        """
        if not target_str:
            return ""
            
        # Tách chuỗi: "người, cái cốc" -> ["người", "cái cốc"]
        targets = [t.strip() for t in target_str.split(',') if t.strip()]
        
        if self.prompt_gen:
            # Dùng generator để ghép vào template có sẵn
            return self.prompt_gen.generate(targets)
        else:
            # Fallback nếu không load được file txt
            joiner = " và "
            return f"hãy tìm {joiner.join(targets)}"

    def preprocess_image(self, frame):
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img).float().to(self.device)

    def image_callback(self, msg):
        if not self.current_prompt:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return

        h_orig, w_orig = frame.shape[:2]

        # 1. Preprocess & Encode Text
        img_tensor = self.preprocess_image(frame)
        text_ids = self.tokenizer.encode(self.current_prompt).unsqueeze(0).to(self.device)

        # 2. Inference
        with torch.no_grad():
            output = self.model(img_tensor, text_ids)
            if isinstance(output, dict): output = output['main']
            prob_map = torch.sigmoid(output)[0, 0]
            mask = (prob_map > self.threshold).cpu().numpy().astype(np.uint8)

        # 3. Post-process
        mask_resized = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        
        # 4. Visualization
        overlay = frame.copy()
        if np.any(mask_resized):
            color_layer = np.zeros_like(frame)
            color_layer[mask_resized > 0] = self.mask_color
            overlay = cv2.addWeighted(frame, 1.0, color_layer, 0.5, 0)
            
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        # In prompt lên ảnh để biết đang tìm cái gì
        cv2.putText(overlay, f"AI Finding: {self.current_prompt}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 5. Publish
        self.pub_mask.publish(self.bridge.cv2_to_imgmsg(mask_resized * 255, "mono8"))
        self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(overlay, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    node = ReferringSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
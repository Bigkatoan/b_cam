import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import os

# ==============================================================================
# 1. MODEL ARCHITECTURE
# ==============================================================================
class _DenseLayer(nn.Module):
    def __init__(self, in_c, growth_rate, bn_size, drop_rate=0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_c)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_c, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_c, bn_size, growth_rate, drop_rate=0):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_c + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module(f'denselayer{i+1}', layer)

class _Transition(nn.Sequential):
    def __init__(self, in_c, out_c, down=True):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_c))
        self.add_module('relu', nn.ReLU(inplace=True))
        if down:
            self.add_module('conv', nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False))
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.add_module('conv', nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))

class DenseUNet(nn.Module):
    def __init__(self, num_classes=1, growth_rate=32, block_config=(4, 4, 4, 4), 
                 num_init_features=64, compression=0.5, drop_rate=0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        c = num_init_features
        self.enc_blocks = nn.ModuleList()
        self.trans_down = nn.ModuleList()
        skips = []
        for i, nl in enumerate(block_config):
            self.enc_blocks.append(_DenseBlock(nl, c, 4, growth_rate, drop_rate))
            c += nl * growth_rate
            if i != len(block_config) - 1:
                self.trans_down.append(_Transition(c, int(c * compression), down=True))
                skips.append(int(c * compression))
                c = int(c * compression)
        self.final_enc_c = c
        c = self.final_enc_c
        self.up3 = _Transition(c, skips[2], down=False)
        self.db_up3 = _DenseBlock(block_config[2], skips[2]*2, 4, growth_rate, drop_rate)
        c = skips[2]*2 + block_config[2]*growth_rate
        self.aux2 = nn.Conv2d(c, num_classes, 1)
        self.reduce2 = nn.Conv2d(c, skips[1], 1)
        self.up2 = _Transition(skips[1], skips[1], down=False)
        self.db_up2 = _DenseBlock(block_config[1], skips[1]*2, 4, growth_rate, drop_rate)
        c = skips[1]*2 + block_config[1]*growth_rate
        self.aux1 = nn.Conv2d(c, num_classes, 1)
        self.reduce1 = nn.Conv2d(c, skips[0], 1)
        self.up1 = _Transition(skips[0], skips[0], down=False)
        self.db_up1 = _DenseBlock(block_config[0], skips[0]*2, 4, growth_rate, drop_rate)
        c = skips[0]*2 + block_config[0]*growth_rate
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(c, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
        
    def forward(self, x):
        x0 = self.features(x)
        x1 = self.enc_blocks[0](x0); s1 = self.trans_down[0](x1)
        x2 = self.enc_blocks[1](s1); s2 = self.trans_down[1](x2)
        x3 = self.enc_blocks[2](s2); s3 = self.trans_down[2](x3)
        x4 = self.enc_blocks[3](s3)
        d3 = self.db_up3(torch.cat([s3, F.interpolate(self.up3(x4), size=s3.shape[2:])], 1))
        d2 = self.db_up2(torch.cat([s2, F.interpolate(self.up2(self.reduce2(d3)), size=s2.shape[2:])], 1))
        d1 = self.db_up1(torch.cat([s1, F.interpolate(self.up1(self.reduce1(d2)), size=s1.shape[2:])], 1))
        out = self.final_up(d1)
        return out

# ==============================================================================
# 2. ROS2 NODE CLASS
# ==============================================================================

class HumanSegmentNode(Node):
    def __init__(self):
        super().__init__('human_segment_node')

        # --- TỰ ĐỘNG TÌM ĐƯỜNG DẪN MODEL ---
        default_model_path = ""
        try:
            # Tìm trong thư mục share/b_cam_ai/weights/ sau khi install
            pkg_share = get_package_share_directory('b_cam_ai')
            default_model_path = os.path.join(pkg_share, 'weights', 'human_aug_ep50.pth')
        except Exception:
            self.get_logger().warn("Cannot find package share directory. Using local path.")
            default_model_path = './weights/human_aug_ep50.pth'
        
        # --- Parameters ---
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('img_size', 256)
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.declare_parameter('input_topic', '/camera/camera/color/image_raw') 

        # Lấy giá trị tham số
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.img_size = self.get_parameter('img_size').get_parameter_value().integer_value
        self.threshold = self.get_parameter('threshold').get_parameter_value().double_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value

        self.get_logger().info(f"Starting Human Segmentation on {self.device}")
        self.get_logger().info(f"Loading weights from: {self.model_path}")
        
        # --- Init Model ---
        self.model = DenseUNet(num_classes=1, drop_rate=0.0).to(self.device)
        self.model.eval()
        self.load_model_weights()
        
        # --- ROS2 Infrastructure ---
        self.bridge = CvBridge()
        
        # Subscriber (QoS Best Effort để tối ưu cho ảnh stream)
        self.sub = self.create_subscription(
            Image, input_topic, self.image_callback, qos_profile_sensor_data
        )
        
        # Publishers
        self.pub_mask = self.create_publisher(Image, 'b_cam_ai/human_mask', 10)
        self.pub_overlay = self.create_publisher(Image, 'b_cam_ai/human_overlay', 10)

    def load_model_weights(self):
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.get_logger().info("Weight loaded successfully!")
            except Exception as e:
                self.get_logger().error(f"Error loading model weights: {e}")
        else:
            self.get_logger().error(f"Model file NOT FOUND at: {self.model_path}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        h_orig, w_orig = frame.shape[:2]

        # Pre-processing
        img_resized = cv2.resize(frame, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = TF.to_tensor(img_rgb).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(img_tensor)
            if isinstance(logits, list): logits = logits[0]
            probs = torch.sigmoid(logits)
            mask = (probs > self.threshold).float()
            
            mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255 
            mask_orig = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # Visualization (Overlay)
        overlay = frame.copy()
        contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2) # Viền xanh lá

        # Publish
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(mask_orig, "mono8")
            mask_msg.header = msg.header
            self.pub_mask.publish(mask_msg)

            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, "bgr8")
            overlay_msg.header = msg.header
            self.pub_overlay.publish(overlay_msg)
            
        except Exception as e:
            self.get_logger().error(f"Publish Error: {e}")

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
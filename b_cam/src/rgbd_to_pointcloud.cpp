#include "b_cam/rgbd_to_pointcloud.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cstring> // Cần thiết cho std::memcpy
#include <limits>  // Cần thiết cho std::numeric_limits

namespace b_cam
{

RgbdToPointCloudNode::RgbdToPointCloudNode(const rclcpp::NodeOptions & options)
: Node("rgbd_to_pointcloud_node", options)
{
    // Tham số cấu hình khoảng cách (mét)
    this->declare_parameter("min_depth", 0.2); // Gần hơn 0.2m thì cắt (do camera mù)
    this->declare_parameter("max_depth", 3.0); // Xa hơn 3m thì cắt (do nhiễu)
    
    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(qos_profile), qos_profile);

    sub_rgb_.subscribe(this, "camera/color/image_raw", qos.get_rmw_qos_profile());
    sub_depth_.subscribe(this, "camera/depth/image_raw", qos.get_rmw_qos_profile());
    sub_info_.subscribe(this, "camera/color/camera_info", qos.get_rmw_qos_profile());

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(5), sub_rgb_, sub_depth_, sub_info_
    );

    sync_->registerCallback(std::bind(&RgbdToPointCloudNode::topic_callback, this, 
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    pub_pointcloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("b_cam/points", 10);
}

void RgbdToPointCloudNode::topic_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info_msg)
{
    double min_d = this->get_parameter("min_depth").as_double();
    double max_d = this->get_parameter("max_depth").as_double();

    cv_bridge::CvImagePtr rgb_ptr, depth_ptr;
    try {
        rgb_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
        depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1); 
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    double fx = info_msg->k[0];
    double fy = info_msg->k[4];
    double cx = info_msg->k[2];
    double cy = info_msg->k[5];
    double depth_scale = 0.001; 

    // --- SỬA TẠI ĐÂY: Dùng biến cloud_msg nhất quán ---
    auto cloud_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    cloud_msg->header = rgb_msg->header; 
    cloud_msg->height = rgb_msg->height;
    cloud_msg->width = rgb_msg->width;
    cloud_msg->is_dense = false;
    cloud_msg->is_bigendian = false;

    // Sử dụng field 'rgb' dạng float thay vì r,g,b rời rạc để tránh lỗi hiển thị
    sensor_msgs::PointCloud2Modifier modifier(*cloud_msg); // Đã sửa từ cloud_out -> cloud_msg
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(rgb_msg->height * rgb_msg->width);

    sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
    // Iterator cho RGB (đóng gói trong float)
    sensor_msgs::PointCloud2Iterator<float> iter_rgb(*cloud_msg, "rgb");

    const cv::Mat& depth_img = depth_ptr->image;
    const cv::Mat& rgb_img = rgb_ptr->image;

    for (int v = 0; v < depth_img.rows; ++v)
    {
        for (int u = 0; u < depth_img.cols; ++u, ++iter_x, ++iter_y, ++iter_z, ++iter_rgb)
        {
            uint16_t depth_raw = depth_img.at<uint16_t>(v, u);
            float z = static_cast<float>(depth_raw) * depth_scale;

            // --- THUẬT TOÁN LỌC KHOẢNG CÁCH ---
            // Nếu z quá xa hoặc quá gần hoặc = 0, gán NaN
            if (depth_raw == 0 || z > max_d || z < min_d)
            {
                *iter_x = *iter_y = *iter_z = std::numeric_limits<float>::quiet_NaN();
            }
            else
            {
                *iter_x = (u - cx) * z / fx;
                *iter_y = (v - cy) * z / fy;
                *iter_z = z;

                // --- XỬ LÝ MÀU CHÍNH XÁC ---
                cv::Vec3b color = rgb_img.at<cv::Vec3b>(v, u);
                // OpenCV là BGR, ROS cần RGB. 
                // Pack màu: 0x00RRGGBB
                uint8_t r = color[2]; 
                uint8_t g = color[1];
                uint8_t b = color[0];
                
                uint32_t rgb_val = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                
                // Copy bit từ uint32 sang float (không phải cast)
                std::memcpy(&(*iter_rgb), &rgb_val, sizeof(uint32_t));
            }
        }
    }
    pub_pointcloud_->publish(std::move(cloud_msg));
}

} // namespace b_cam

// Đăng ký component (nếu cần) hoặc chạy dynamic
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(b_cam::RgbdToPointCloudNode)

// Hàm main để chạy node độc lập
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<b_cam::RgbdToPointCloudNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
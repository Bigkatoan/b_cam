#include "b_cam/semantic_fusion_node.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace b_cam
{

SemanticFusionNode::SemanticFusionNode(const rclcpp::NodeOptions & options)
: Node("semantic_fusion_node", options)
{
    // QoS Best Effort cho dữ liệu sensor
    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(qos_profile), qos_profile);

    // Subscribe
    // 1. PointCloud (nên dùng bản downsampled cho nhanh)
    sub_cloud_.subscribe(this, "b_cam/points_downsampled", qos.get_rmw_qos_profile());
    // 2. Mask từ AI Node
    sub_mask_.subscribe(this, "b_cam_ai/human_mask", qos.get_rmw_qos_profile());
    // 3. Camera Info để lấy ma trận K
    sub_info_.subscribe(this, "camera/color/camera_info", qos.get_rmw_qos_profile());

    // Sync Policy: Queue size 20 để chờ AI xử lý xong frame
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(20), sub_cloud_, sub_mask_, sub_info_
    );

    sync_->registerCallback(std::bind(&SemanticFusionNode::topic_callback, this, 
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    // Output
    pub_human_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("b_cam/human_points", 10);
    pub_static_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("b_cam/static_points", 10);

    RCLCPP_INFO(this->get_logger(), "Semantic Fusion Node Started: Mixing 3D & AI.");
}

void SemanticFusionNode::topic_callback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& mask_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info_msg)
{
    // 1. Convert Mask sang OpenCV Mat
    cv::Mat mask;
    try {
        mask = cv_bridge::toCvCopy(mask_msg, "mono8")->image;
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
        return;
    }

    // 2. Lấy tham số Camera (Pinhole Model)
    // K = [fx 0 cx; 0 fy cy; 0 0 1]
    double fx = info_msg->k[0];
    double fy = info_msg->k[4];
    double cx = info_msg->k[2];
    double cy = info_msg->k[5];

    // 3. Chuẩn bị Output Clouds
    auto human_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    auto static_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();

    // Copy Header (Frame ID và Timestamp rất quan trọng)
    human_msg->header = cloud_msg->header;
    static_msg->header = cloud_msg->header;
    
    // Setup modifier để resize cloud
    sensor_msgs::PointCloud2Modifier mod_human(*human_msg);
    sensor_msgs::PointCloud2Modifier mod_static(*static_msg);
    
    mod_human.setPointCloud2FieldsByString(2, "xyz", "rgb");
    mod_static.setPointCloud2FieldsByString(2, "xyz", "rgb");

    // Ta chưa biết kích thước cuối cùng, resize tạm bằng input rồi shrink sau
    // Hoặc dùng vector trung gian. Ở đây để tối ưu ta dùng vector trung gian lưu index.
    std::vector<int> human_indices;
    std::vector<int> static_indices;
    human_indices.reserve(cloud_msg->width * cloud_msg->height);
    static_indices.reserve(cloud_msg->width * cloud_msg->height);

    // 4. Duyệt PointCloud Input
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud_msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud_msg, "z");
    
    // Check nếu có màu
    bool has_rgb = false;
    for (const auto& field : cloud_msg->fields) if (field.name == "rgb") has_rgb = true;
    sensor_msgs::PointCloud2ConstIterator<float> iter_rgb(*cloud_msg, "rgb");

    int idx = 0;
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++idx)
    {
        float x = *iter_x;
        float y = *iter_y;
        float z = *iter_z;

        if (std::isnan(x) || std::isnan(z) || z <= 0) {
            if (has_rgb) ++iter_rgb;
            continue;
        }

        // --- PROJECTION 3D -> 2D ---
        // u = fx * (x/z) + cx
        // v = fy * (y/z) + cy
        int u = static_cast<int>(fx * (x / z) + cx);
        int v = static_cast<int>(fy * (y / z) + cy);

        // Check xem điểm chiếu có nằm trong ảnh không
        if (u >= 0 && u < mask.cols && v >= 0 && v < mask.rows)
        {
            // Kiểm tra giá trị pixel tại Mask (255 = Người, 0 = Nền)
            // Lưu ý: AI có thể mask không hoàn hảo, ta có thể erode/dilate nếu cần
            if (mask.at<uint8_t>(v, u) > 128) {
                human_indices.push_back(idx);
            } else {
                static_indices.push_back(idx);
            }
        } else {
            // Điểm nằm ngoài khung hình camera (FOV) nhưng vẫn trong cloud (ít gặp với RGBD)
            static_indices.push_back(idx);
        }

        if (has_rgb) ++iter_rgb;
    }

    // 5. Fill dữ liệu vào Output
    mod_human.resize(human_indices.size());
    mod_static.resize(static_indices.size());

    // Helper lambda để copy data
    auto copy_points = [&](sensor_msgs::msg::PointCloud2& out_msg, const std::vector<int>& indices) {
        sensor_msgs::PointCloud2Iterator<float> out_x(out_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> out_y(out_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> out_z(out_msg, "z");
        sensor_msgs::PointCloud2Iterator<float> out_rgb(out_msg, "rgb");

        // Reset iterators của input để đọc lại
        sensor_msgs::PointCloud2ConstIterator<float> in_x(*cloud_msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> in_y(*cloud_msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> in_z(*cloud_msg, "z");
        sensor_msgs::PointCloud2ConstIterator<float> in_rgb(*cloud_msg, "rgb");

        int current_idx = 0;
        int list_ptr = 0;
        
        // Loop tối ưu: Duyệt input 1 lần, nếu index trùng với list thì copy
        // Vì indices đã được push theo thứ tự tăng dần, ta không cần find
        while (in_x != in_x.end() && list_ptr < (int)indices.size())
        {
            if (current_idx == indices[list_ptr]) {
                *out_x = *in_x;
                *out_y = *in_y;
                *out_z = *in_z;
                if (has_rgb) *out_rgb = *in_rgb;
                
                ++out_x; ++out_y; ++out_z; ++out_rgb;
                list_ptr++;
            }
            ++in_x; ++in_y; ++in_z; 
            if (has_rgb) ++in_rgb;
            current_idx++;
        }
    };

    copy_points(*human_msg, human_indices);
    copy_points(*static_msg, static_indices);

    // 6. Publish
    pub_human_cloud_->publish(std::move(human_msg));
    pub_static_cloud_->publish(std::move(static_msg));
}

} // namespace b_cam

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(b_cam::SemanticFusionNode)

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<b_cam::SemanticFusionNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
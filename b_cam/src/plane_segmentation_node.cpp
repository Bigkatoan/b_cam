#include "b_cam/plane_segmentation_node.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cmath>
#include <set>

namespace b_cam
{

PlaneSegmentationNode::PlaneSegmentationNode(const rclcpp::NodeOptions & options)
: Node("plane_segmentation_node", options)
{
    // Cấu hình: 0.02m (2cm) là độ dày chấp nhận được của mặt bàn
    this->declare_parameter("distance_threshold", 0.02);
    this->declare_parameter("max_iterations", 100);

    distance_threshold_ = this->get_parameter("distance_threshold").as_double();
    max_iterations_ = this->get_parameter("max_iterations").as_int();

    // Lắng nghe topic đã downsampled (để chạy cho nhanh)
    sub_points_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "b_cam/points_downsampled", rclcpp::SensorDataQoS(),
        std::bind(&PlaneSegmentationNode::cloud_callback, this, std::placeholders::_1));

    pub_plane_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("b_cam/plane", 10);
    pub_objects_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("b_cam/objects", 10);
    
    RCLCPP_INFO(this->get_logger(), "Plane Segmentation Node Started.");
}

void PlaneSegmentationNode::cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    // 1. Parse dữ liệu từ message sang vector struct để dễ truy cập index
    std::vector<Point3D> points;
    points.reserve(msg->width * msg->height);

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
    
    // Kiểm tra màu
    bool has_rgb = false;
    for (const auto& field : msg->fields) { if (field.name == "rgb") has_rgb = true; }
    sensor_msgs::PointCloud2ConstIterator<float> iter_rgb(*msg, "rgb");

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        Point3D p;
        p.x = *iter_x; p.y = *iter_y; p.z = *iter_z;
        if (has_rgb) {
            p.rgb = *iter_rgb;
            ++iter_rgb;
        } else {
            p.rgb = 0.0f;
        }
        points.push_back(p);
    }

    size_t n_points = points.size();
    if (n_points < 3) return; // Không đủ điểm dựng mặt phẳng

    // 2. Thuật toán RANSAC
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, n_points - 1);

    std::vector<size_t> best_inliers;
    
    for (int i = 0; i < max_iterations_; ++i)
    {
        // Chọn 3 điểm ngẫu nhiên
        size_t idx1 = dis(gen);
        size_t idx2 = dis(gen);
        size_t idx3 = dis(gen);

        Point3D p1 = points[idx1];
        Point3D p2 = points[idx2];
        Point3D p3 = points[idx3];

        // Tạo 2 vector: v1 = p2 - p1, v2 = p3 - p1
        float v1x = p2.x - p1.x; float v1y = p2.y - p1.y; float v1z = p2.z - p1.z;
        float v2x = p3.x - p1.x; float v2y = p3.y - p1.y; float v2z = p3.z - p1.z;

        // Tích có hướng (Cross Product) để tìm vector pháp tuyến (a, b, c)
        float a = v1y * v2z - v1z * v2y;
        float b = v1z * v2x - v1x * v2z;
        float c = v1x * v2y - v1y * v2x;

        // Chuẩn hóa vector pháp tuyến (để tính khoảng cách chính xác)
        float length = std::sqrt(a*a + b*b + c*c);
        if (length == 0) continue; // 3 điểm thẳng hàng, bỏ qua
        a /= length; b /= length; c /= length;

        // Tính d trong phương trình mặt phẳng: ax + by + cz + d = 0 => d = -(ax + by + cz)
        float d = -(a * p1.x + b * p1.y + c * p1.z);

        // Đếm inliers
        std::vector<size_t> current_inliers;
        for (size_t j = 0; j < n_points; ++j) {
            // Khoảng cách từ điểm đến mặt phẳng
            float dist = std::abs(a * points[j].x + b * points[j].y + c * points[j].z + d);
            if (dist <= distance_threshold_) {
                current_inliers.push_back(j);
            }
        }

        // Lưu lại kết quả tốt nhất
        if (current_inliers.size() > best_inliers.size()) {
            best_inliers = current_inliers;
        }
    }

    // 3. Tách Cloud thành 2 phần: Plane và Objects
    auto plane_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    auto objects_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    
    // Cấu hình header giống nhau
    plane_msg->header = msg->header; plane_msg->height = 1; 
    plane_msg->is_dense = true; plane_msg->is_bigendian = false;
    objects_msg->header = msg->header; objects_msg->height = 1;
    objects_msg->is_dense = true; objects_msg->is_bigendian = false;

    sensor_msgs::PointCloud2Modifier mod_plane(*plane_msg);
    sensor_msgs::PointCloud2Modifier mod_obj(*objects_msg);
    
    mod_plane.setPointCloud2FieldsByString(2, "xyz", "rgb");
    mod_obj.setPointCloud2FieldsByString(2, "xyz", "rgb");

    mod_plane.resize(best_inliers.size());
    mod_obj.resize(n_points - best_inliers.size());

    // Iterators cho Plane
    sensor_msgs::PointCloud2Iterator<float> out_p_x(*plane_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> out_p_y(*plane_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> out_p_z(*plane_msg, "z");
    sensor_msgs::PointCloud2Iterator<float> out_p_rgb(*plane_msg, "rgb");

    // Iterators cho Objects
    sensor_msgs::PointCloud2Iterator<float> out_o_x(*objects_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> out_o_y(*objects_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> out_o_z(*objects_msg, "z");
    sensor_msgs::PointCloud2Iterator<float> out_o_rgb(*objects_msg, "rgb");

    // Dùng Set để tra cứu nhanh inliers
    std::set<size_t> inlier_set(best_inliers.begin(), best_inliers.end());

    for (size_t i = 0; i < n_points; ++i) {
        const auto& p = points[i];
        if (inlier_set.count(i)) {
            // Là mặt phẳng -> Ghi vào plane_msg
            *out_p_x = p.x; *out_p_y = p.y; *out_p_z = p.z; *out_p_rgb = p.rgb;
            ++out_p_x; ++out_p_y; ++out_p_z; ++out_p_rgb;
        } else {
            // Là vật thể -> Ghi vào objects_msg
            *out_o_x = p.x; *out_o_y = p.y; *out_o_z = p.z; *out_o_rgb = p.rgb;
            ++out_o_x; ++out_o_y; ++out_o_z; ++out_o_rgb;
        }
    }

    pub_plane_->publish(std::move(plane_msg));
    pub_objects_->publish(std::move(objects_msg));
}

} // namespace b_cam

// Helper Main
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(b_cam::PlaneSegmentationNode)

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<b_cam::PlaneSegmentationNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
#ifndef B_CAM__PLANE_SEGMENTATION_NODE_HPP_
#define B_CAM__PLANE_SEGMENTATION_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vector>
#include <random>

namespace b_cam
{

struct Point3D {
    float x, y, z;
    // Helper để đọc màu (packed float)
    float rgb; 
};

class PlaneSegmentationNode : public rclcpp::Node
{
public:
    explicit PlaneSegmentationNode(const rclcpp::NodeOptions & options);

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_points_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_plane_;   // Xuất ra mặt bàn
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_objects_; // Xuất ra vật thể

    // Tham số RANSAC
    double distance_threshold_; // Độ dày của mặt phẳng (mét)
    int max_iterations_;        // Số lần thử ngẫu nhiên

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
};

} // namespace b_cam

#endif // B_CAM__PLANE_SEGMENTATION_NODE_HPP_
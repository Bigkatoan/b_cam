#ifndef B_CAM__VOXEL_GRID_NODE_HPP_
#define B_CAM__VOXEL_GRID_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vector>
#include <unordered_map>

namespace b_cam
{

// Cấu trúc để lưu trữ thông tin cộng dồn của một Voxel
struct VoxelData {
    float x_sum = 0.0f;
    float y_sum = 0.0f;
    float z_sum = 0.0f;
    int r_sum = 0;
    int g_sum = 0;
    int b_sum = 0;
    int count = 0;
};

class VoxelGridNode : public rclcpp::Node
{
public:
    explicit VoxelGridNode(const rclcpp::NodeOptions & options);

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_points_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_filtered_;

    // Tham số kích thước voxel (mét)
    double leaf_size_; 

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
};

} // namespace b_cam

#endif // B_CAM__VOXEL_GRID_NODE_HPP_
#ifndef B_CAM__ODOM_NODE_HPP_
#define B_CAM__ODOM_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

// SOTA PCL Headers
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h> // Generalized ICP (Robust hơn ICP thường)
#include <pcl/filters/voxel_grid.h> // Để quản lý Local Map

#include <deque>
#include <mutex>
#include <Eigen/Dense>

namespace b_cam
{

struct KeyFrame {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    Eigen::Matrix4f pose;
    double time;
};

class OdomNode : public rclcpp::Node
{
public:
    explicit OdomNode(const rclcpp::NodeOptions & options);

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_voxel_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_local_map_; // Debug Map

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // Core Algorithms
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp_;
    pcl::VoxelGrid<pcl::PointXYZ> map_downsampler_;

    // State Variables
    Eigen::Matrix4f global_pose_;
    Eigen::Matrix4f t_base_optical_;
    
    // Local Map Management (Sliding Window)
    std::deque<KeyFrame> keyframes_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_map_;
    
    // Config
    int max_keyframes_;         // Số lượng frame trong submap (VD: 20)
    double keyframe_delta_dist_; // Khoảng cách tối thiểu để thêm keyframe

    std::mutex odom_mutex_;

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void update_local_map(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Eigen::Matrix4f& pose);
    void publish_odom(const rclcpp::Time& stamp);
};

} // namespace b_cam

#endif // B_CAM__ODOM_NODE_HPP_
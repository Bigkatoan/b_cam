#include "b_cam/odom_node.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace b_cam
{

OdomNode::OdomNode(const rclcpp::NodeOptions & options)
: Node("odom_node", options),
  local_map_(new pcl::PointCloud<pcl::PointXYZ>()),
  global_pose_(Eigen::Matrix4f::Identity())
{
    // --- PARAMETERS ---
    this->declare_parameter("max_local_map_size", 20); // Giữ 20 frame gần nhất làm map
    this->declare_parameter("keyframe_delta", 0.3);    // 0.3m di chuyển thì update map
    
    max_keyframes_ = this->get_parameter("max_local_map_size").as_int();
    keyframe_delta_dist_ = this->get_parameter("keyframe_delta").as_double();

    // --- SOTA GICP CONFIGURATION ---
    // GICP mạnh hơn ICP vì nó mô hình hóa bề mặt cục bộ (covariance)
    gicp_.setMaxCorrespondenceDistance(0.2); // Tìm điểm khớp trong 20cm
    gicp_.setMaximumIterations(30);          // Real-time limit
    gicp_.setTransformationEpsilon(1e-6);
    gicp_.setEuclideanFitnessEpsilon(1e-6);
    // Tùy chọn: Dùng RANSAC nội bộ của GICP nếu môi trường quá nhiễu (tốn CPU hơn)
    // gicp_.setRANSACIterations(10); 

    // Downsampler cho Map (để map không quá nặng)
    map_downsampler_.setLeafSize(0.1, 0.1, 0.1); // Map thưa hơn input một chút

    // --- SUBSCRIBERS / PUBLISHERS ---
    // QoS Best Effort để ưu tiên tốc độ mới nhất
    sub_voxel_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "b_cam/points_downsampled", rclcpp::SensorDataQoS(),
        std::bind(&OdomNode::cloud_callback, this, std::placeholders::_1));

    pub_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("b_cam/odom", 10);
    pub_local_map_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("b_cam/local_map", 1);
    
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // --- TF: OPTICAL TO BASE ---
    Eigen::AngleAxisf rollAngle(-M_PI / 2, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf yawAngle(-M_PI / 2, Eigen::Vector3f::UnitZ());
    Eigen::Quaternionf q = yawAngle * rollAngle;
    t_base_optical_ = Eigen::Matrix4f::Identity();
    t_base_optical_.block<3,3>(0,0) = q.toRotationMatrix();

    RCLCPP_INFO(this->get_logger(), "SOTA GICP Odometry (Scan-to-Local-Map) Started.");
}

void OdomNode::cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(odom_mutex_);

    // 1. Convert & Transform to Base Frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *cloud_in);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*cloud_in, *current_cloud, t_base_optical_);

    if (current_cloud->size() < 50) return;

    // 2. Initialization
    if (keyframes_.empty()) {
        update_local_map(current_cloud, global_pose_);
        return;
    }

    // 3. SCAN-TO-LOCAL-MAP MATCHING
    // Thay vì so với prev_cloud, ta so với local_map_ (ổn định hơn nhiều)
    gicp_.setInputSource(current_cloud);
    gicp_.setInputTarget(local_map_);

    // Guess pose: Dùng pose cũ làm điểm bắt đầu
    // (Trong thực tế nên dùng Motion Model: Pose_t = Pose_t-1 + Velocity * dt)
    // Ở đây dùng Pose_t-1 là đủ cho chuyển động chậm
    Eigen::Matrix4f guess = global_pose_; 
    
    pcl::PointCloud<pcl::PointXYZ> unused_result;
    gicp_.align(unused_result, guess);

    if (gicp_.hasConverged())
    {
        // GICP trả về trực tiếp Global Pose mới (do ta align với Map Global)
        // Không cần nhân nghịch đảo delta như Frame-to-Frame
        global_pose_ = gicp_.getFinalTransformation();

        // 4. Update Map Strategy
        // Chỉ thêm keyframe mới vào bản đồ nếu đã di chuyển đủ xa
        KeyFrame& last_kf = keyframes_.back();
        double dist = (global_pose_.block<3,1>(0,3) - last_kf.pose.block<3,1>(0,3)).norm();

        if (dist > keyframe_delta_dist_) {
            update_local_map(current_cloud, global_pose_);
        }

        publish_odom(msg->header.stamp);
    }
    else {
        RCLCPP_WARN(this->get_logger(), "GICP Lost Track! Motion too fast or dynamic environment.");
    }
}

void OdomNode::update_local_map(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Eigen::Matrix4f& pose)
{
    // Thêm Keyframe mới
    KeyFrame kf;
    kf.pose = pose;
    // Lưu cloud đã được transform về đúng vị trí Global của nó
    kf.cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*cloud, *kf.cloud, pose); 
    keyframes_.push_back(kf);

    // Xóa Keyframe cũ nếu quá limit (Sliding Window)
    if (keyframes_.size() > (size_t)max_keyframes_) {
        keyframes_.pop_front();
    }

    // Rebuild Local Map từ các Keyframe
    local_map_->clear();
    for (const auto& frame : keyframes_) {
        *local_map_ += *frame.cloud;
    }

    // Downsample Map để GICP chạy nhanh (Map càng lớn càng chậm)
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_filtered(new pcl::PointCloud<pcl::PointXYZ>());
    map_downsampler_.setInputCloud(local_map_);
    map_downsampler_.filter(*map_filtered);
    local_map_ = map_filtered;

    // Publish Map để debug trên RViz
    sensor_msgs::msg::PointCloud2 map_msg;
    pcl::toROSMsg(*local_map_, map_msg);
    map_msg.header.frame_id = "odom";
    map_msg.header.stamp = this->now();
    pub_local_map_->publish(map_msg);
}

void OdomNode::publish_odom(const rclcpp::Time& stamp)
{
    float x = global_pose_(0, 3);
    float y = global_pose_(1, 3);
    float z = global_pose_(2, 3);

    Eigen::Matrix3f rot = global_pose_.block<3, 3>(0, 0);
    Eigen::Quaternionf q(rot);

    // TF Broadcaster
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = stamp;
    t.header.frame_id = "odom";
    t.child_frame_id = "base_link";

    t.transform.translation.x = x;
    t.transform.translation.y = y;
    t.transform.translation.z = z;
    t.transform.rotation.x = q.x();
    t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z();
    t.transform.rotation.w = q.w();
    tf_broadcaster_->sendTransform(t);

    // Odometry Msg
    auto odom_msg = std::make_unique<nav_msgs::msg::Odometry>();
    odom_msg->header.stamp = stamp;
    odom_msg->header.frame_id = "odom";
    odom_msg->child_frame_id = "base_link";
    odom_msg->pose.pose.position.x = x;
    odom_msg->pose.pose.position.y = y;
    odom_msg->pose.pose.position.z = z;
    odom_msg->pose.pose.orientation = t.transform.rotation;
    
    // Covariance giả định (quan trọng nếu dùng EKF sau này)
    // Nếu GICP fitness score thấp -> Covariance nhỏ (tin tưởng)
    double score = gicp_.getFitnessScore();
    odom_msg->pose.covariance[0] = score * 100; // x var
    odom_msg->pose.covariance[7] = score * 100; // y var
    
    pub_odom_->publish(std::move(odom_msg));
}

} // namespace b_cam

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(b_cam::OdomNode)

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<b_cam::OdomNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
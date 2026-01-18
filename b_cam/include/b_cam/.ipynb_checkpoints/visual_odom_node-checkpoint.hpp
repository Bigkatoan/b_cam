#ifndef B_CAM__VISUAL_ODOM_NODE_HPP_
#define B_CAM__VISUAL_ODOM_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

namespace b_cam
{

class VisualOdomNode : public rclcpp::Node
{
public:
    explicit VisualOdomNode(const rclcpp::NodeOptions & options);

private:
    // Policy đồng bộ: Cho phép trễ 1 chút để khớp RGB và Depth
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image,
        sensor_msgs::msg::CameraInfo
    >;

    message_filters::Subscriber<sensor_msgs::msg::Image> sub_rgb_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_depth_;
    message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_info_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // Dữ liệu Frame trước
    cv::Mat prev_gray_;
    cv::Mat prev_depth_;
    std::vector<cv::Point2f> prev_keypoints_;

    // Pose toàn cục của Robot (Base Frame)
    cv::Mat global_pose_; 
    
    // Ma trận chuyển đổi từ Optical Frame (Z-tới) sang Base Frame (X-tới)
    cv::Mat t_base_optical_; 

    void topic_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info_msg
    );
    
    void publish_odom(const rclcpp::Time& stamp);
};

} // namespace b_cam

#endif // B_CAM__VISUAL_ODOM_NODE_HPP_
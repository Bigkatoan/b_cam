#ifndef B_CAM__RGBD_TO_POINTCLOUD_HPP_
#define B_CAM__RGBD_TO_POINTCLOUD_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

namespace b_cam
{

class RgbdToPointCloudNode : public rclcpp::Node
{
public:
    explicit RgbdToPointCloudNode(const rclcpp::NodeOptions & options);

private:
    // Định nghĩa policy để đồng bộ hóa các topic (Approximate Time là quan trọng nhất cho camera thực)
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image,
        sensor_msgs::msg::CameraInfo
    >;

    // Subscribers
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_rgb_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_depth_;
    message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_info_;
    
    // Synchronizer
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // Publisher
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_pointcloud_;

    // Callback xử lý chính
    void topic_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info_msg
    );
};

} // namespace b_cam

#endif // B_CAM__RGBD_TO_POINTCLOUD_HPP_
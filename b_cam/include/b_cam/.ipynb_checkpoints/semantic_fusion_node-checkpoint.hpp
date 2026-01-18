#ifndef B_CAM__SEMANTIC_FUSION_NODE_HPP_
#define B_CAM__SEMANTIC_FUSION_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

namespace b_cam
{

class SemanticFusionNode : public rclcpp::Node
{
public:
    explicit SemanticFusionNode(const rclcpp::NodeOptions & options);

private:
    // Policy đồng bộ: Cần độ trễ lớn hơn chút vì AI chạy chậm hơn PointCloud
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2,
        sensor_msgs::msg::Image,
        sensor_msgs::msg::CameraInfo
    >;

    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_mask_;
    message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_info_;
    
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_human_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_static_cloud_;

    void topic_callback(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& mask_msg,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info_msg
    );
};

} // namespace b_cam

#endif
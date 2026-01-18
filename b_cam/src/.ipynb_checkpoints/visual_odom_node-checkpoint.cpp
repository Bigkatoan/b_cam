#include "b_cam/visual_odom_node.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <opencv2/video/tracking.hpp>

namespace b_cam
{

VisualOdomNode::VisualOdomNode(const rclcpp::NodeOptions & options)
: Node("visual_odom_node", options)
{
    // --- PARAMETERS ---
    // Cắt bỏ độ sâu xa quá mức cho phép (User yêu cầu < 2.8m)
    // Tôi để mặc định 2.5m cho an toàn tuyệt đối
    this->declare_parameter("min_depth", 0.2); 
    this->declare_parameter("max_depth", 2.5); 
    this->declare_parameter("min_inliers", 15); // Cần ít nhất 15 điểm tốt mới tính Odom

    // QoS Setup
    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(qos_profile), qos_profile);

    sub_rgb_.subscribe(this, "camera/color/image_raw", qos.get_rmw_qos_profile());
    sub_depth_.subscribe(this, "camera/depth/image_raw", qos.get_rmw_qos_profile());
    sub_info_.subscribe(this, "camera/color/camera_info", qos.get_rmw_qos_profile());

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), sub_rgb_, sub_depth_, sub_info_
    );

    sync_->registerCallback(std::bind(&VisualOdomNode::topic_callback, this, 
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    pub_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("b_cam/odom", 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Khởi tạo Pose = Identity
    global_pose_ = cv::Mat::eye(4, 4, CV_64F);

    // Ma trận xoay Optical -> Base (Z-fwd -> X-fwd)
    // Base X = Opt Z | Base Y = -Opt X | Base Z = -Opt Y
    t_base_optical_ = cv::Mat::zeros(3, 3, CV_64F);
    t_base_optical_.at<double>(0, 2) = 1;  
    t_base_optical_.at<double>(1, 0) = -1; 
    t_base_optical_.at<double>(2, 1) = -1; 

    RCLCPP_INFO(this->get_logger(), "Visual Odom Started. Max Depth Cutoff: %.2fm", 
        this->get_parameter("max_depth").as_double());
}

void VisualOdomNode::topic_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info_msg)
{
    // Lấy params
    double min_d_m = this->get_parameter("min_depth").as_double();
    double max_d_m = this->get_parameter("max_depth").as_double();
    int min_inliers = this->get_parameter("min_inliers").as_int();

    // 1. Convert Image
    cv::Mat curr_gray, curr_depth;
    try {
        cv::Mat rgb = cv_bridge::toCvCopy(rgb_msg, "bgr8")->image;
        cv::cvtColor(rgb, curr_gray, cv::COLOR_BGR2GRAY);
        // Depth image thường là 16-bit unsigned (mm)
        curr_depth = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
    } catch (...) { return; }

    // Ma trận nội tham số Camera K
    cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
    K.at<double>(0,0) = info_msg->k[0]; K.at<double>(0,2) = info_msg->k[2];
    K.at<double>(1,1) = info_msg->k[4]; K.at<double>(1,2) = info_msg->k[5];
    K.at<double>(2,2) = 1.0;

    std::vector<cv::Point2f> curr_keypoints;
    std::vector<cv::Point3f> prev_pts_3d_valid;
    std::vector<cv::Point2f> curr_pts_2d_valid;

    // Detect lại nếu mất dấu hoặc chưa có keypoints
    if (prev_keypoints_.size() < 50) {
        // GoodFeaturesToTrack tốt cho Optical Flow hơn ORB
        cv::goodFeaturesToTrack(curr_gray, curr_keypoints, 200, 0.01, 10);
    } 
    else {
        // Optical Flow Tracking (Lucas-Kanade)
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_gray_, curr_gray, prev_keypoints_, curr_keypoints, status, err);

        // Lọc điểm
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
                cv::Point2f pt_prev = prev_keypoints_[i];
                cv::Point2f pt_curr = curr_keypoints[i];
                
                // --- BỘ LỌC ĐỘ SÂU NGHIÊM NGẶT ---
                int u = (int)pt_prev.x; 
                int v = (int)pt_prev.y;
                
                // Check biên
                if (u < 0 || u >= prev_depth_.cols || v < 0 || v >= prev_depth_.rows) continue;
                
                uint16_t d_raw = prev_depth_.at<uint16_t>(v, u);
                float d_metric = d_raw * 0.001f; // Chuyển sang mét

                // QUAN TRỌNG: Loại bỏ ngay lập tức nếu ngoài vùng tin cậy (2.5m)
                // Đây là bước khử nhiễu chính cho Realsense
                if (d_raw == 0 || d_metric < min_d_m || d_metric > max_d_m) {
                    continue; 
                }

                // Nếu điểm hợp lệ, tính toạ độ 3D cũ (Back-projection)
                float z = d_metric;
                float x = (u - K.at<double>(0,2)) * z / K.at<double>(0,0);
                float y = (v - K.at<double>(1,2)) * z / K.at<double>(1,1);

                prev_pts_3d_valid.emplace_back(x, y, z);
                curr_pts_2d_valid.push_back(pt_curr);
            }
        }
    }

    // --- PnP Solver (Visual Odometry Core) ---
    if (prev_pts_3d_valid.size() >= (size_t)min_inliers) {
        cv::Mat rvec, tvec, R;
        std::vector<int> inliers;
        
        // RANSAC chặt chẽ: Reprojection error = 2.0 pixel (Mặc định 8.0 rất lỏng lẻo)
        // Giúp loại bỏ điểm bị trôi
        bool success = cv::solvePnPRansac(prev_pts_3d_valid, curr_pts_2d_valid, K, cv::noArray(), 
                                          rvec, tvec, false, 100, 2.0, 0.99, inliers);

        if (success && inliers.size() > (size_t)min_inliers) {
            double dist = cv::norm(tvec);
            
            // --- BỘ LỌC CHUYỂN ĐỘNG ---
            // 1. Chặn nhảy xa bất thường (> 0.5m/frame là vô lý)
            // 2. Chặn nhiễu đứng yên (< 1mm/frame thì coi như đứng yên để khử rung/drift)
            if (dist < 0.5 && dist > 0.001) { 
                cv::Rodrigues(rvec, R);
                
                // Tạo ma trận Transformation 4x4
                cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
                R.copyTo(T(cv::Rect(0,0,3,3)));
                tvec.copyTo(T(cv::Rect(3,0,1,3)));

                // Pose Camera mới = Pose Cũ * T.inv()
                cv::Mat T_inv = T.inv(); // Camera Motion in Optical Frame

                // Chuyển Motion từ Optical Frame sang Base Frame (Robot)
                // Công thức: T_base = R_bo * T_opt * R_bo.T
                cv::Mat R_bo = t_base_optical_;
                cv::Mat R_opt_motion = T_inv(cv::Rect(0,0,3,3));
                cv::Mat t_opt_motion = T_inv(cv::Rect(3,0,1,3));

                cv::Mat R_base_motion = R_bo * R_opt_motion * R_bo.t();
                cv::Mat t_base_motion = R_bo * t_opt_motion;

                cv::Mat T_base_step = cv::Mat::eye(4, 4, CV_64F);
                R_base_motion.copyTo(T_base_step(cv::Rect(0,0,3,3)));
                t_base_motion.copyTo(T_base_step(cv::Rect(3,0,1,3)));

                // Cộng dồn vào Global Pose
                global_pose_ = global_pose_ * T_base_step;
                
                publish_odom(rgb_msg->header.stamp);
            }
        }
    }

    // Cập nhật dữ liệu cho vòng lặp sau
    // Nếu số điểm track được quá ít (< 30), xóa hết để frame sau detect lại từ đầu
    if (curr_pts_2d_valid.size() < 30) {
        prev_keypoints_.clear(); 
    } else {
        prev_keypoints_ = curr_pts_2d_valid;
    }
    
    prev_gray_ = curr_gray;
    prev_depth_ = curr_depth;
}

void VisualOdomNode::publish_odom(const rclcpp::Time& stamp)
{
    double tx = global_pose_.at<double>(0, 3);
    double ty = global_pose_.at<double>(1, 3);
    double tz = global_pose_.at<double>(2, 3);

    cv::Mat R = global_pose_(cv::Rect(0,0,3,3));
    tf2::Matrix3x3 tf_rot(
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2)
    );
    tf2::Quaternion q;
    tf_rot.getRotation(q);

    // TF Broadcasting (Quan trọng để các node khác hiển thị đúng)
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = stamp;
    t.header.frame_id = "odom";
    t.child_frame_id = "base_link";
    t.transform.translation.x = tx;
    t.transform.translation.y = ty;
    t.transform.translation.z = tz;
    t.transform.rotation = tf2::toMsg(q);
    tf_broadcaster_->sendTransform(t);

    // Odom Message
    nav_msgs::msg::Odometry odom;
    odom.header = t.header;
    odom.child_frame_id = t.child_frame_id;
    odom.pose.pose.position.x = tx;
    odom.pose.pose.position.y = ty;
    odom.pose.pose.position.z = tz;
    odom.pose.pose.orientation = t.transform.rotation;
    
    // Gán Covariance nhỏ (tin tưởng cao)
    odom.pose.covariance[0] = 0.0001; 
    pub_odom_->publish(odom);
}

} // namespace b_cam

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(b_cam::VisualOdomNode)

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<b_cam::VisualOdomNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
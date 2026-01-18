from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # ---------------------------------------------------------
        # Node 1: Pre-processing (RGBD -> PointCloud)
        # ---------------------------------------------------------
        Node(
            package='b_cam',
            executable='rgbd_to_pointcloud_node',
            name='b_cam_realsense_processor',
            output='screen',
            parameters=[
                {'min_depth': 0.2}, 
                {'max_depth': 2.5} # Cắt xa 2.5m để loại bỏ nhiễu nặng của Realsense
            ],
            remappings=[
                ('camera/color/image_raw', '/camera/camera/color/image_raw'),
                ('camera/depth/image_raw', '/camera/camera/aligned_depth_to_color/image_raw'),
                ('camera/color/camera_info', '/camera/camera/color/camera_info')
            ]
        ),
        
        # ---------------------------------------------------------
        # Node 2: Voxel Grid (Giảm tải dữ liệu)
        # ---------------------------------------------------------
        Node(
            package='b_cam',
            executable='voxel_grid_node',
            name='b_cam_voxel_filter',
            output='screen',
            parameters=[
                {'leaf_size': 0.05},          # 5cm
                {'min_points_per_voxel': 10}  # Lọc nhiễu lấm tấm
            ]
        ),
        
        # ---------------------------------------------------------
        # Node 3: Plane Segmentation (Tách bàn/sàn)
        # ---------------------------------------------------------
        Node(
            package='b_cam',
            executable='plane_segmentation_node',
            name='b_cam_plane_segmentation',
            output='screen',
            parameters=[
                {'distance_threshold': 0.02}, # Dung sai mặt phẳng 2cm
                {'max_iterations': 100}
            ]
            # Output topic:
            # - /b_cam/plane (Mặt bàn/Sàn) -> Màu đỏ
            # - /b_cam/objects (Vật thể) -> Màu xanh/Màu thật
        ),

        Node(
            package='b_cam_ai',
            executable='human_segment_node',
            name='human_segment_node',
            output='screen',
            parameters=[
                {'img_size': 256},      # Resize ảnh input model
                {'threshold': 0.5},     # Ngưỡng chọn mask
                {'input_topic': '/camera/camera/color/image_raw'} # Input là ảnh màu gốc
            ]
            # Output topics: 
            # - /b_cam_ai/human_mask
            # - /b_cam_ai/human_overlay
        ),

        Node(
            package='b_cam',
            executable='semantic_fusion_node',
            name='b_cam_semantic_fusion',
            output='screen',
            remappings=[
                # Input 1: PointCloud đã giảm tải (Downsampled)
                ('b_cam/points_downsampled', '/b_cam/points_downsampled'),
                
                # Input 2: Mask người từ AI Node
                ('b_cam_ai/human_mask', '/b_cam_ai/human_mask'),
                
                # Input 3: Thông số Camera để chiếu điểm
                ('camera/color/camera_info', '/camera/camera/color/camera_info')
            ]
            # Output:
            # - /b_cam/human_points (Chỉ người)
            # - /b_cam/static_points (Chỉ môi trường - Đã xóa người)
        )
    ])
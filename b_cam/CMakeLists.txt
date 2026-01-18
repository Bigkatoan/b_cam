cmake_minimum_required(VERSION 3.8)
project(b_cam)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(include)

# Node 1: RGBD to PointCloud
add_executable(rgbd_to_pointcloud_node src/rgbd_to_pointcloud.cpp)
ament_target_dependencies(rgbd_to_pointcloud_node
  rclcpp rclcpp_components sensor_msgs cv_bridge message_filters image_transport OpenCV
)

# Node 2: Voxel Grid
add_executable(voxel_grid_node src/voxel_grid_node.cpp)
ament_target_dependencies(voxel_grid_node
  rclcpp rclcpp_components sensor_msgs
)

# Node 3: Plane Segmentation
add_executable(plane_segmentation_node src/plane_segmentation_node.cpp)
ament_target_dependencies(plane_segmentation_node
  rclcpp rclcpp_components sensor_msgs
)

add_executable(semantic_fusion_node src/semantic_fusion_node.cpp)
ament_target_dependencies(semantic_fusion_node
  rclcpp rclcpp_components sensor_msgs cv_bridge message_filters OpenCV
)

install(TARGETS
  rgbd_to_pointcloud_node
  voxel_grid_node
  plane_segmentation_node
  semantic_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
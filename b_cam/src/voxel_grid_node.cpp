#include "b_cam/voxel_grid_node.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace b_cam
{

struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHasher {
    std::size_t operator()(const VoxelKey& k) const {
        return ((std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1)) >> 1) ^ (std::hash<int>()(k.z) << 1);
    }
};

VoxelGridNode::VoxelGridNode(const rclcpp::NodeOptions & options)
: Node("voxel_grid_node", options)
{
    this->declare_parameter("leaf_size", 0.05);
    this->declare_parameter("min_points_per_voxel", 5); // Tối thiểu 5 điểm mới tạo thành 1 voxel (Giảm nhiễu bay lơ lửng)
    
    leaf_size_ = this->get_parameter("leaf_size").as_double();

    sub_points_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "b_cam/points", rclcpp::SensorDataQoS(), 
        std::bind(&VoxelGridNode::cloud_callback, this, std::placeholders::_1));

    pub_filtered_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("b_cam/points_downsampled", 10);
}

void VoxelGridNode::cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    if (msg->data.empty()) return;
    int min_points = this->get_parameter("min_points_per_voxel").as_int();

    std::unordered_map<VoxelKey, VoxelData, VoxelKeyHasher> grid;

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
    
    // Kiểm tra xem có màu không để xử lý
    bool has_rgb = false;
    for (const auto& field : msg->fields) { if (field.name == "rgb") has_rgb = true; }

    // Iterator đọc RGB (dạng float packed)
    sensor_msgs::PointCloud2ConstIterator<float> iter_rgb(*msg, "rgb");

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
    {
        float x = *iter_x; float y = *iter_y; float z = *iter_z;

        if (std::isnan(x) || std::isnan(y) || std::isnan(z)) {
            if (has_rgb) ++iter_rgb; 
            continue;
        }

        VoxelKey key;
        key.x = static_cast<int>(std::floor(x / leaf_size_));
        key.y = static_cast<int>(std::floor(y / leaf_size_));
        key.z = static_cast<int>(std::floor(z / leaf_size_));

        VoxelData& voxel = grid[key];
        voxel.x_sum += x;
        voxel.y_sum += y;
        voxel.z_sum += z;
        voxel.count++;

        if (has_rgb) {
            // Unpack màu từ float -> uint32 -> r,g,b
            uint32_t rgb_val;
            std::memcpy(&rgb_val, &(*iter_rgb), sizeof(uint32_t));
            
            // Format: 0x00RRGGBB
            uint8_t r = (rgb_val >> 16) & 0x0000ff;
            uint8_t g = (rgb_val >> 8)  & 0x0000ff;
            uint8_t b = (rgb_val)       & 0x0000ff;

            voxel.r_sum += r;
            voxel.g_sum += g;
            voxel.b_sum += b;
            ++iter_rgb;
        }
    }

    // --- LỌC BỎ VOXEL NHIỄU (quá ít điểm) ---
    // Tạo cloud output
    auto cloud_out = std::make_unique<sensor_msgs::msg::PointCloud2>();
    cloud_out->header = msg->header;
    cloud_out->height = 1;
    cloud_out->is_dense = true;
    cloud_out->is_bigendian = false;
    
    sensor_msgs::PointCloud2Modifier modifier(*cloud_out);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    
    // Chỉ resize bằng số lượng voxel đủ điều kiện
    // (Ta không biết trước, nên dùng reserve và push_back hoặc resize max rồi shrink sau, 
    // nhưng để tối ưu code, ta sẽ loop 2 lần hoặc dùng Vector tạm. Ở đây dùng modifier resize dần thì chậm.
    // Cách tốt nhất: đếm trước).
    size_t valid_voxels = 0;
    for (auto const& [key, val] : grid) {
        if (val.count >= min_points) valid_voxels++;
    }
    modifier.resize(valid_voxels);

    sensor_msgs::PointCloud2Iterator<float> out_x(*cloud_out, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y(*cloud_out, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z(*cloud_out, "z");
    sensor_msgs::PointCloud2Iterator<float> out_rgb(*cloud_out, "rgb");

    for (auto it = grid.begin(); it != grid.end(); ++it)
    {
        const VoxelData& v = it->second;
        
        // Bỏ qua nếu voxel quá thưa (nhiễu)
        if (v.count < min_points) continue;

        float count_f = static_cast<float>(v.count);

        *out_x = v.x_sum / count_f;
        *out_y = v.y_sum / count_f;
        *out_z = v.z_sum / count_f;

        if (has_rgb) {
            uint8_t r = static_cast<uint8_t>(v.r_sum / v.count);
            uint8_t g = static_cast<uint8_t>(v.g_sum / v.count);
            uint8_t b = static_cast<uint8_t>(v.b_sum / v.count);
            
            // Re-pack lại màu
            uint32_t rgb_val = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
            std::memcpy(&(*out_rgb), &rgb_val, sizeof(uint32_t));
        } else {
            // Mặc định trắng
            uint32_t white = 0x00FFFFFF;
            std::memcpy(&(*out_rgb), &white, sizeof(uint32_t));
        }

        ++out_x; ++out_y; ++out_z; ++out_rgb;
    }

    pub_filtered_->publish(std::move(cloud_out));
}

} // namespace b_cam

// Main function
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<b_cam::VoxelGridNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
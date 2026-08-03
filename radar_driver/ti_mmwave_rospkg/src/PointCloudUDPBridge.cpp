#include "PointCloudUDPBridge.h"
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <string>
#include <unistd.h>
#include <vector>

#pragma pack(push, 1)
struct PointShort {
  int16_t xs;
  int16_t ys;
  int16_t zs;
};
#pragma pack(pop)

PointCloudUdpBridge::PointCloudUdpBridge() : Node("pointcloud_udp_bridge") {
  udp_sock_ = socket(AF_INET, SOCK_DGRAM, 0);
  if (udp_sock_ < 0) {
    RCLCPP_FATAL(this->get_logger(), "Failed to create UDP socket");
    throw std::runtime_error("Cannot create socket");
  }

  std::memset(&broadcast_addr_, 0, sizeof(broadcast_addr_));
  broadcast_addr_.sin_family = AF_INET;
  broadcast_addr_.sin_port = htons(4200); // change this if needed
  inet_pton(AF_INET, "239.255.0.1", &broadcast_addr_.sin_addr);

  unsigned char ttl = 1;
  struct in_addr localInterface;
  localInterface.s_addr = inet_addr("127.0.0.1");
  if (setsockopt(udp_sock_, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)) <
      0) {
    RCLCPP_WARN(
        this->get_logger(),
        "Failed to set IP_MULTICAST_TTL. Defaulting to system behavior.");
  }
  if (setsockopt(udp_sock_, IPPROTO_IP, IP_MULTICAST_IF,
                 (char *)&localInterface, sizeof(localInterface)) < 0) {
    RCLCPP_WARN(
        this->get_logger(),
        "Failed to set IP_MULTICAST_IF. Defaulting to system behavior.");
  }

  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/ti_mmwave/radar_scan_pcl", 10,
      std::bind(&PointCloudUdpBridge::pointcloud_callback, this,
                std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(), "UDP Server Initialized at %s:%d",
              "239.255.0.1", 4200);
}

PointCloudUdpBridge::~PointCloudUdpBridge() {
  if (udp_sock_ >= 0) {
    close(udp_sock_);
  }
}

void PointCloudUdpBridge::pointcloud_callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

  std::vector<PointShort> points;
  points.reserve(512);

  for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
    const float x = *iter_x;
    const float y = *iter_y;
    const float z = *iter_z;

    PointShort point;
    std::int16_t x_s16 = static_cast<int16_t>(x * 1000);
    std::int16_t y_s16 = static_cast<int16_t>(y * 1000);
    std::int16_t z_s16 = static_cast<int16_t>(z * 1000);

    point.xs = x_s16;
    point.ys = y_s16;
    point.zs = z_s16;
    points.emplace_back(point);
  }

  uint32_t timestamp_us = this->get_clock()->now().nanoseconds() / 1000;
  uint16_t payload_size = points.size() * sizeof(PointShort);
  // header | timestamp | pointcloud data | footer
  size_t packet_size = 2 + sizeof(timestamp_us) + sizeof(payload_size) + payload_size + 2;
  std::vector<std::byte> packet(packet_size);
  RCLCPP_INFO(this->get_logger(), "Pointcloud Size: %d. Payload Size:%ld", payload_size, packet_size);

  size_t index = 0;

  packet.at(0) = std::byte{':'};
  packet.at(1) = std::byte{':'};
  index = 2;
  std::memcpy(packet.data() + index, &timestamp_us, sizeof(timestamp_us));
  index += sizeof(timestamp_us);
  std::memcpy(packet.data() + index, &payload_size, sizeof(payload_size));
  index += sizeof(payload_size);
  std::memcpy(packet.data() + index, points.data(), payload_size);
  index += payload_size;
  packet.at(index) = std::byte{';'};
  packet.at(index + 1) = std::byte{';'};

  ssize_t sent =
      sendto(udp_sock_, packet.data(), packet.size(), 0,
             (struct sockaddr *)&broadcast_addr_, sizeof(broadcast_addr_));
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointCloudUdpBridge>());
  rclcpp::shutdown();
  return 0;
}
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <sys/socket.h>
#include <arpa/inet.h>


class PointCloudUdpBridge : public rclcpp::Node {
public:
    PointCloudUdpBridge() ;

    ~PointCloudUdpBridge();

private:
  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);


  int udp_sock_ = -1;
  struct sockaddr_in broadcast_addr_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
};
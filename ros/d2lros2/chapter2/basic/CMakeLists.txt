cmake_minimum_required(VERSION 3.22)

project(first_ros2_node)

find_package(rclcpp REQUIRED)
add_executable(first_node first_ros2_node.cpp)

target_link_libraries(first_node rclcpp::rclcpp)
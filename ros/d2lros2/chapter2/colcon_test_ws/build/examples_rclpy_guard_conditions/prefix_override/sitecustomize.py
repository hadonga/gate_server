import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/config/workspace/ros/d2lros2/chapter2/colcon_test_ws/install/examples_rclpy_guard_conditions'

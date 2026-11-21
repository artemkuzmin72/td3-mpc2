import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/artem/ros2_ws/src/turtlebot_rl/install/turtlebot_rl'

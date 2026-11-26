# First terminal - create world
1. cd /home/artem/ros2_ws
2. source /opt/ros/humble/setup.bash
3. source install/setup.bash
4. colcon build
5. ros2 launch turtlebot_rl maze.launch.py 

# Second terminal - start learning
1. cd /home/artem/ros2_ws
2. source /opt/ros/humble/setup.bash
3. source install/setup.bash
4. colcon build
5. ros2 launch turtlebot_rl train_node.py

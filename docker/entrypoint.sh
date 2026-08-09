#!/bin/bash
set -e

# Source the ROS 2 Jazzy underlay
source /opt/ros/jazzy/setup.bash

# Source your custom workspace overlay
# (Only if the setup file exists, which it will after building)
if [ -f /ros2_ws/install/setup.bash ]; then
  source /ros2_ws/install/setup.bash
fi

# Execute the command passed to the container
exec "$@"
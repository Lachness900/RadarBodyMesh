FROM ros:jazzy-ros-base

ENV ROS_WS=/ros2_ws
WORKDIR $ROS_WS

# Install colcon build tools
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY ../radar_driver $ROS_WS/src/radar_driver 

# Install dependencies from package.xml using rosdep
# Note: rosdep is already initialized in the official ros base images
RUN apt-get update && rosdep update && rosdep install -y \
    --from-paths src \
    --ignore-src \
    --rosdistro jazzy \
    && rm -rf /var/lib/apt/lists/*

# Build the ROS 2 workspace
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && colcon --log-path /dev/null build --symlink-install"

# Copy the entrypoint script and ensure it is executable
COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh

# Set the entrypoint to our script
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
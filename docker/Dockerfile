
FROM ros:jazzy-perception

ARG USERNAME=USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3-pip
ENV SHELL /bin/bash

ENV ROS_WS=/home/ros2_ws
WORKDIR $ROS_WS

# Install colcon build tools
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    build-essential \
    ros-jazzy-pcl-conversions \
    ros-jazzy-pcl-ros \
    && rm -rf /var/lib/apt/lists/*


RUN mkdir -p $ROS_WS/src/

COPY radar_driver/ $ROS_WS/src/ 

# Install dependencies from package.xml using rosdep
# Note: rosdep is already initialized in the official ros base images
RUN apt-get update && rosdep update && rosdep install -y \
    --from-paths src \
    --ignore-src \
    --rosdistro jazzy \
    && rm -rf /var/lib/apt/lists/*

# Build the ROS 2 workspace
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && colcon --log-base /dev/null build --symlink-install"

# Copy the entrypoint script and ensure it is executable
COPY ./entrypoint.sh /
RUN chmod +x /entrypoint.sh

# Set the entrypoint to our script
ENTRYPOINT ["/entrypoint.sh"]

# Default command (can be overridden in docker-compose)
USER $USERNAME
CMD ["bash"]

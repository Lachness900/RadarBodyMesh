@echo off
start "Visualiser" cmd "python point_visualizer/visualizer_with_classifier.py -m pose_classifier.pt -f cam_radar_t_pose_1_3.dat"
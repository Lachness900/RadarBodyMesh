## AI for mmYoga
Run in the following order:
- python batch_extract.py --dataset dataset/ --out combined_dataset.npz (Extracts and combines radar data from .dat files into a single file)
- python prepare_pointcloud_dataset.py --in combined_dataset.npz --out prepared_pointcloud_dataset.npz --val_frac 0.2 (Prepares combined data into training/training and sets contraints)
- python train_pointcloud_classifier.py --in prepared_pointcloud_dataset.npz --out pointcloud_classifier.pt --epochs 240 (Trains actual AI and provides final accuracies and confusion matrix)
- TODO python visualizer_with_classifier.py -f test.dat -m pose_classifier.pt (Runs AI model with visualizer on some .dat input for testing)
- python evaluate_pointcloud_test_dataset.py --data combined_testset.npz --model pointcloud_classifier.pt --out_dir eval_results (Takes in batch output of testset folder)
(Note that test.dat is currently the dynamic poses by dominion12 which are not a part of the trained dataset)

dataset is the folder where you save the .dat files used to train the AI.
Expected input layout:
    dataset/
        t_pose/
            cam_radar_t_pose_1_1.dat
            cam_radar_t_pose_1_2.dat
            ...
        standing_pose/
            ...
        warrior_1_pose/
        warrior_2_pose/
        angle_pose/
        other/              <- misc / negative-class captures
            ...
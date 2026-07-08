export const WS_URL =
  import.meta.env.VITE_MMYOGA_WS_URL || "ws://localhost:8000/ws/predictions";

export const POSES = [
  { key: "t_pose", label: "T Pose", color: "#0f8b8d" },
  { key: "straight_pose", label: "Straight Pose", color: "#2f6fbb" },
  { key: "warrior_pose", label: "Warrior Pose", color: "#e4572e" },
  { key: "other_pose", label: "Other Pose", color: "#6c5b7b" },
];

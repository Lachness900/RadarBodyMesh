export const WS_URL =
  import.meta.env.VITE_MMYOGA_WS_URL || "ws://localhost:8000/ws/predictions";
export const API_URL = import.meta.env.VITE_MMYOGA_API_URL || "http://localhost:8000";

export function buildPredictionWsUrl({ source, replayFile }) {
  const url = new URL(WS_URL);
  url.searchParams.set("source", source || "auto");
  if (replayFile) {
    url.searchParams.set("replay_file", replayFile);
  }
  return url.toString();
}

export const POSES = [
  { key: "t_pose", label: "T Pose", color: "#1ae1e4" },
  { key: "standing_pose", label: "Standing Pose", color: "#2f6fbb" },
  { key: "warrior_1_pose", label: "Right Warrior Pose", color: "#e4572e" },
  { key: "warrior_2_pose", label: "Left Warrior Pose", color: "#7ae42e" },
  { key: "angle_pose", label: "Angle Pose", color: "#f1ea16" },
  { key: "other", label: "Other Pose", color: "#6c5b7b" },
];

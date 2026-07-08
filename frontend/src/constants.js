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
  { key: "t_pose", label: "T Pose", color: "#0f8b8d" },
  { key: "straight_pose", label: "Straight Pose", color: "#2f6fbb" },
  { key: "warrior_pose", label: "Warrior Pose", color: "#e4572e" },
  { key: "other_pose", label: "Other Pose", color: "#6c5b7b" },
];

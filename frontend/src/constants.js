export const WS_URL =
  import.meta.env.VITE_MMYOGA_WS_URL || "ws://localhost:8000/ws/predictions";
export const API_URL = import.meta.env.VITE_MMYOGA_API_URL || "http://localhost:8000";
const REFERENCE_POSE_ROOT = `${import.meta.env.BASE_URL}reference-poses`;
const REFERENCE_POSE_VERSION = "20260806-squat";

function referencePoseAsset(filename) {
  return `${REFERENCE_POSE_ROOT}/${filename}?v=${REFERENCE_POSE_VERSION}`;
}

export function buildPredictionWsUrl({ source, replayFile }) {
  const url = new URL(WS_URL);
  url.searchParams.set("source", source || "auto");
  if (replayFile) {
    url.searchParams.set("replay_file", replayFile);
  }
  return url.toString();
}

export const POSES = [
  { key: "standing_pose", label: "Standing Pose", color: "#2f6fbb" },
  { key: "t_pose", label: "T Pose", color: "#1ae1e4" },
  { key: "squat", label: "Squat Pose", color: "#e4572e" },
  { key: "angle_pose", label: "Angle Pose", color: "#f1ea16" },
];

export const REFERENCE_POSES = {
  standing_pose: referencePoseAsset("standing_pose_male.glb"),
  t_pose: referencePoseAsset("t_pose_male.glb"),
  squat: referencePoseAsset("squat_male.glb"),
  angle_pose: referencePoseAsset("angle_pose_male.glb"),
};

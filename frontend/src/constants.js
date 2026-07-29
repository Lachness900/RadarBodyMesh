export const WS_URL =
  import.meta.env.VITE_MMYOGA_WS_URL || "ws://localhost:8000/ws/predictions";
export const API_URL = import.meta.env.VITE_MMYOGA_API_URL || "http://localhost:8000";
const REFERENCE_POSE_ROOT = `${import.meta.env.BASE_URL}reference-poses`;
const REFERENCE_POSE_VERSION = "20260729-warrior-1-prayer-joined";

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
  { key: "t_pose", label: "T Pose", color: "#1ae1e4" },
  { key: "standing_pose", label: "Standing Pose", color: "#2f6fbb" },
  { key: "warrior_1_pose", label: "Warrior Pose 1", color: "#e4572e" },
  { key: "warrior_2_pose", label: "Warrior Pose 2", color: "#7ae42e" },
  { key: "angle_pose", label: "Angle Pose", color: "#f1ea16" },
  { key: "other", label: "Other Pose", color: "#6c5b7b" },
];

// "other" deliberately remains unmapped because it is a catch-all class, not a
// single pose that could be represented honestly by one reference mesh.
export const REFERENCE_POSES = {
  t_pose: referencePoseAsset("t_pose_male.glb"),
  standing_pose: referencePoseAsset("standing_pose_male.glb"),
  warrior_1_pose: referencePoseAsset("warrior_1_pose_male.glb"),
  warrior_2_pose: referencePoseAsset("warrior_2_pose_male.glb"),
  angle_pose: referencePoseAsset("angle_pose_male.glb"),
};

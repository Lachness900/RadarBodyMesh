import { backendOriginFromPage } from "./network";

const PAGE_URL =
  typeof window === "undefined" ? "http://localhost:5173" : window.location.href;
const PAGE_IS_SECURE = new URL(PAGE_URL).protocol === "https:";
const DEFAULT_API_URL = backendOriginFromPage(
  PAGE_URL,
  PAGE_IS_SECURE ? "https:" : "http:",
);
const DEFAULT_WS_URL = `${backendOriginFromPage(
  PAGE_URL,
  PAGE_IS_SECURE ? "wss:" : "ws:",
)}/ws/predictions`;

export const WS_URL =
  import.meta.env.VITE_MMPOSE_WS_URL ||
  import.meta.env.VITE_MMYOGA_WS_URL ||
  DEFAULT_WS_URL;
export const API_URL =
  import.meta.env.VITE_MMPOSE_API_URL ||
  import.meta.env.VITE_MMYOGA_API_URL ||
  DEFAULT_API_URL;
const REFERENCE_POSE_ROOT = `${import.meta.env.BASE_URL}reference-poses`;
const REFERENCE_POSE_VERSION = "20260808-squat-v14";

function referencePoseAsset(filename) {
  return `${REFERENCE_POSE_ROOT}/${filename}?v=${REFERENCE_POSE_VERSION}`;
}

export function buildPredictionWsUrl({ source, replayFile, model }) {
  const url = new URL(WS_URL);
  url.searchParams.set("source", source || "auto");
  if (replayFile) {
    url.searchParams.set("replay_file", replayFile);
  }
  if (source !== "mock" && model) {
    url.searchParams.set("model", model);
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

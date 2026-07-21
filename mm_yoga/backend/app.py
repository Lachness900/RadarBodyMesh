"""FastAPI backend for mmYoga replay inference and frontend WebSocket updates."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by runtime setup
    raise SystemExit(
        "FastAPI is required for the backend. Install dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc

from mm_yoga.backend.replay import (
    async_mock_messages,
    async_replay_messages,
    iter_replay_messages,
    mock_message,
)
from mm_yoga.model.inference import load_predictor

DEFAULT_REPLAY_FILE = Path(
    "OneDrive/DepthCam_Radar_Cloud_Combined/cam_radar_1783260788477740516.dat"
)
UPLOAD_REPLAY_DIR = Path("data/replay/uploads")
DEFAULT_REPLAY_DIRS = [
    Path("OneDrive/DepthCam_Radar_Cloud_Combined"),
    UPLOAD_REPLAY_DIR,
]

# The backend defaults to replay mode because the first dashboard milestone is
# about showing an existing recording, not connecting live radar yet.
app = FastAPI(title="mmYoga backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _model_path() -> Path | None:
    """Optional future model file path.

    If this is unset, ``load_predictor`` returns the deterministic mock
    predictor. Real model loading is intentionally not part of this stage.
    """

    raw_path = "pose_classifier.pt"
    return Path(raw_path) if raw_path else None


def _replay_path() -> Path:
    """Recording used by REST/WebSocket replay endpoints."""

    return Path(os.getenv("MMYOGA_REPLAY_FILE", str(DEFAULT_REPLAY_FILE)))


def _available_replay_files() -> list[Path]:
    """Known replay recordings that use the synchronized cam/radar packet format."""

    files: dict[str, Path] = {}
    default_path = _replay_path()
    if default_path.exists():
        files[str(default_path)] = default_path
    for directory in DEFAULT_REPLAY_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.dat")):
            files[str(path)] = path
    return list(files.values())


def _selected_replay_path(raw_path: str | None = None) -> Path:
    """Resolve a selected replay file path from the frontend."""

    if not raw_path:
        return _replay_path()
    return Path(raw_path).expanduser()


def _safe_upload_name(raw_name: str | None) -> str:
    """Return a local filename safe for the replay upload directory."""

    name = Path(raw_name or "uploaded_replay.dat").name
    if not name.lower().endswith(".dat"):
        name = f"{name}.dat"
    return name


def _source_mode(raw_source: str | None) -> str:
    """Normalize source query values used by REST and WebSocket endpoints."""

    source = (raw_source or "auto").lower()
    return source if source in {"auto", "replay", "mock"} else "auto"


def _mock_latest() -> dict[str, object]:
    """Return one mock message with the same shape as replay messages."""

    return mock_message(timestamp_ms=0.0)


@app.get("/health")
def health() -> dict[str, object]:
    replay_path = _replay_path()
    return {
        "status": "ok",
        "replay_file": str(replay_path),
        "replay_available": replay_path.exists(),
        "model_file": str(_model_path()) if _model_path() else None,
    }


@app.get("/api/sources")
def sources() -> dict[str, object]:
    replay_files = _available_replay_files()
    default_path = _replay_path()
    return {
        "default_source": "replay" if default_path.exists() else "mock",
        "default_replay_file": str(default_path),
        "sources": [
            {"id": "mock", "label": "Mock"},
            {"id": "replay", "label": "Replay"},
        ],
        "replay_files": [
            {
                "path": str(path),
                "label": path.name,
                "selected": path == default_path,
            }
            for path in replay_files
        ],
    }


@app.post("/api/replay-files")
async def upload_replay_file(request: Request, filename: str | None = None) -> dict[str, object]:
    """Save an uploaded local ``.dat`` file for replay mode.

    The frontend uploads the file body as ``application/octet-stream`` so this
    endpoint does not need the optional python-multipart dependency.
    """

    UPLOAD_REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    output_path = UPLOAD_REPLAY_DIR / _safe_upload_name(filename)
    with output_path.open("wb") as file_obj:
        async for chunk in request.stream():
            if chunk:
                file_obj.write(chunk)
    return {
        "path": str(output_path),
        "label": output_path.name,
        "selected": True,
    }


@app.get("/api/latest")
def latest(source: str = "auto", replay_file: str | None = None) -> dict[str, object]:
    mode = _source_mode(source)
    if mode == "mock":
        return _mock_latest()

    predictor = load_predictor(_model_path())
    replay_path = _selected_replay_path(replay_file)
    if replay_path.exists():
        for message in iter_replay_messages(replay_path, predictor=predictor, max_frames=1):
            return message
    # Keep the REST endpoint usable on machines that do not have replay data.
    return _mock_latest()


@app.websocket("/ws/predictions")
async def predictions(websocket: WebSocket) -> None:
    await websocket.accept()
    predictor = load_predictor(_model_path())
    mode = _source_mode(websocket.query_params.get("source"))
    replay_path = _selected_replay_path(websocket.query_params.get("replay_file"))
    try:
        if mode == "mock":
            async for message in async_mock_messages():
                await websocket.send_json(message)
        elif replay_path.exists():
            # Re-open the finite replay file after it reaches EOF so a dashboard
            # can stay connected during demos.
            while True:
                async for message in async_replay_messages(
                    replay_path,
                    predictor=predictor,
                    playback_speed=float(os.getenv("MMYOGA_PLAYBACK_SPEED", "1.0")),
                ):
                    await websocket.send_json(message)
        else:
            # Development fallback when the recording is not present locally.
            async for message in async_mock_messages():
                await websocket.send_json(message)
    except (RuntimeError, WebSocketDisconnect):
        return

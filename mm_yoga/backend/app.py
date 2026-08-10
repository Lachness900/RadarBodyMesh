"""FastAPI backend for mmPose radar inference and frontend WebSocket updates."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
import logging
import os
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by runtime setup
    raise SystemExit(
        "FastAPI is required for the backend. Install dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc

from mm_yoga.backend.live import LiveRadarService, LiveRadarUnavailable
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


@dataclass(frozen=True)
class ModelOption:
    """One server-approved checkpoint exposed to the dashboard."""

    id: str
    label: str
    path: Path


BUILT_IN_MODELS = (
    ModelOption("pointcloud_240", "PointNet 240", Path("pointcloud_classifier240.pt")),
    ModelOption("pointcloud_120", "PointNet 120", Path("pointcloud_classifier120f.pt")),
)
LEGACY_MODEL = ModelOption("legacy_raster", "Legacy raster", Path("pose_classifier.pt"))
UPLOAD_REPLAY_DIR = Path("data/replay/uploads")
DEFAULT_REPLAY_DIRS = [
    Path("OneDrive/DepthCam_Radar_Cloud_Combined"),
    UPLOAD_REPLAY_DIR,
]
logger = logging.getLogger("uvicorn.error")


def _environment_value(primary: str, legacy: str, default: str | None = None) -> str | None:
    """Read the mmPose variable while keeping legacy launch scripts working."""

    return os.getenv(primary, os.getenv(legacy, default))


LIVE_RADAR_HOST = _environment_value(
    "MMPOSE_RADAR_HOST", "MMYOGA_RADAR_HOST", "239.255.0.1"
)
LIVE_RADAR_PORT = int(
    _environment_value("MMPOSE_RADAR_PORT", "MMYOGA_RADAR_PORT", "4200")
)
LIVE_RADAR_INTERFACE = _environment_value(
    "MMPOSE_RADAR_INTERFACE", "MMYOGA_RADAR_INTERFACE", "127.0.0.1"
)
LIVE_RADAR_TIMEOUT_S = float(
    _environment_value("MMPOSE_RADAR_TIMEOUT", "MMYOGA_RADAR_TIMEOUT", "5.0")
)
live_radar = LiveRadarService(
    host=LIVE_RADAR_HOST,
    port=LIVE_RADAR_PORT,
    interface=LIVE_RADAR_INTERFACE,
    receive_timeout_s=LIVE_RADAR_TIMEOUT_S,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Release the UDP receiver thread when FastAPI shuts down."""

    try:
        yield
    finally:
        live_radar.stop()


# Replay remains the deterministic default; Live is selected explicitly.
app = FastAPI(title="mmPose backend", version="0.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _same_path(left: Path, right: Path) -> bool:
    """Compare configured and built-in paths without requiring either to exist."""

    return left.expanduser().resolve(strict=False) == right.resolve(strict=False)


def _available_model_options() -> list[ModelOption]:
    """Return the fixed set of checkpoints the frontend may request."""

    options = [option for option in BUILT_IN_MODELS if option.path.exists()]
    raw_path = _environment_value("MMPOSE_MODEL_FILE", "MMYOGA_MODEL_FILE")
    if raw_path:
        configured_path = Path(raw_path).expanduser()
        if not any(_same_path(configured_path, option.path) for option in options):
            options.insert(
                0,
                ModelOption("configured", configured_path.stem, configured_path),
            )
    if not options and LEGACY_MODEL.path.exists():
        options.append(LEGACY_MODEL)
    return options


def _default_model_option() -> ModelOption:
    """Resolve the configured/default model from the server whitelist."""

    options = _available_model_options()
    if not options:
        # Preserve a useful error path when checkpoints have not been downloaded.
        return BUILT_IN_MODELS[0]
    raw_path = _environment_value("MMPOSE_MODEL_FILE", "MMYOGA_MODEL_FILE")
    if raw_path:
        configured_path = Path(raw_path).expanduser()
        for option in options:
            if _same_path(configured_path, option.path):
                return option
    return options[0]


def _selected_model(raw_id: str | None = None) -> ModelOption:
    """Resolve a requested model ID without accepting arbitrary file paths."""

    if not raw_id:
        return _default_model_option()
    options = {option.id: option for option in _available_model_options()}
    try:
        return options[raw_id]
    except KeyError as exc:
        raise ValueError(f"Unknown or unavailable model: {raw_id}") from exc


def _model_path() -> Path:
    """Return the default checkpoint path for health and compatibility output."""

    return _default_model_option().path


@lru_cache(maxsize=4)
def _cached_predictor(path: str):
    """Load each selectable checkpoint once instead of on every connection."""

    return load_predictor(Path(path))


def _replay_path() -> Path:
    """Recording used by REST/WebSocket replay endpoints."""

    return Path(
        _environment_value(
            "MMPOSE_REPLAY_FILE",
            "MMYOGA_REPLAY_FILE",
            str(DEFAULT_REPLAY_FILE),
        )
    )


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
    return source if source in {"live", "replay", "mock"} else "auto"


def _mock_latest() -> dict[str, object]:
    """Return one mock message with the same shape as replay messages."""

    return mock_message(timestamp_ms=0.0)


@app.get("/health")
def health() -> dict[str, object]:
    replay_path = _replay_path()
    default_model = _default_model_option()
    return {
        "status": "ok",
        "replay_file": str(replay_path),
        "replay_available": replay_path.exists(),
        "model_id": default_model.id,
        "model_file": str(default_model.path),
        "live": live_radar.status(),
    }


@app.get("/api/sources")
def sources() -> dict[str, object]:
    replay_files = _available_replay_files()
    default_path = _replay_path()
    live_status = live_radar.status()
    model_options = _available_model_options()
    default_model = _default_model_option()
    return {
        "default_source": "replay" if default_path.exists() else "mock",
        "default_replay_file": str(default_path),
        "sources": [
            {"id": "mock", "label": "Mock", "enabled": True},
            {"id": "replay", "label": "Replay", "enabled": True},
            {
                "id": "live",
                "label": "Live Radar",
                "enabled": True,
                "status": live_status,
            },
        ],
        "replay_files": [
            {
                "path": str(path),
                "label": path.name,
                "selected": path == default_path,
            }
            for path in replay_files
        ],
        "default_model": default_model.id,
        "models": [
            {
                "id": option.id,
                "label": option.label,
                "selected": option.id == default_model.id,
            }
            for option in model_options
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


@app.post("/api/live-model")
def select_live_model(model: str) -> dict[str, object]:
    """Change the one model shared by every Live Radar dashboard client."""

    try:
        model_option = _selected_model(model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    predictor = _cached_predictor(str(model_option.path))
    changed = live_radar.configure_model(
        model_id=model_option.id,
        predictor=predictor,
    )
    return {
        "model_id": model_option.id,
        "changed": changed,
        "live": live_radar.status(),
    }


@app.get("/api/latest")
def latest(
    source: str = "auto",
    replay_file: str | None = None,
    model: str | None = None,
) -> dict[str, object]:
    mode = _source_mode(source)
    if mode == "mock":
        return _mock_latest()
    if mode == "live":
        try:
            model_option = _selected_model(live_radar.model_id or model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        message = live_radar.latest_message
        if message is not None and message.get("model_id") == model_option.id:
            return message
        raise HTTPException(
            status_code=503,
            detail={"message": "No live radar frame is available yet", **live_radar.status()},
        )

    try:
        model_option = _selected_model(model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    predictor = _cached_predictor(str(model_option.path))
    replay_path = _selected_replay_path(replay_file)
    if replay_path.exists():
        for message in iter_replay_messages(
            replay_path,
            predictor=predictor,
            model_id=model_option.id,
            max_frames=1,
        ):
            return message
    # Keep the REST endpoint usable on machines that do not have replay data.
    return _mock_latest()


@app.websocket("/ws/predictions")
async def predictions(websocket: WebSocket) -> None:
    await websocket.accept()
    mode = _source_mode(websocket.query_params.get("source"))
    replay_path = _selected_replay_path(websocket.query_params.get("replay_file"))
    try:
        if mode == "mock":
            async for message in async_mock_messages():
                await websocket.send_json(message)
        elif mode == "live":
            # Joining Live observes the current global model. The query value is
            # used only to initialise a stream that has not selected one yet.
            model_option = _selected_model(
                live_radar.model_id or websocket.query_params.get("model")
            )
            predictor = _cached_predictor(str(model_option.path))
            live_radar.start(
                loop=asyncio.get_running_loop(),
                model_id=model_option.id,
                predictor=predictor,
            )
            async for message in live_radar.messages():
                await websocket.send_json(message)
        elif replay_path.exists():
            model_option = _selected_model(websocket.query_params.get("model"))
            predictor = _cached_predictor(str(model_option.path))
            # Re-open the finite replay file after it reaches EOF so a dashboard
            # can stay connected during demos.
            while True:
                async for message in async_replay_messages(
                    replay_path,
                    predictor=predictor,
                    model_id=model_option.id,
                    playback_speed=float(
                        _environment_value(
                            "MMPOSE_PLAYBACK_SPEED",
                            "MMYOGA_PLAYBACK_SPEED",
                            "1.0",
                        )
                    ),
                ):
                    await websocket.send_json(message)
        elif mode == "replay":
            await websocket.close(code=1008, reason="Selected replay file was not found")
        else:
            # Development fallback when the recording is not present locally.
            async for message in async_mock_messages():
                await websocket.send_json(message)
    except ValueError as exc:
        try:
            await websocket.close(code=1008, reason=str(exc)[:120])
        except RuntimeError:
            pass
    except FileNotFoundError as exc:
        logger.error("Model checkpoint unavailable: %s", exc)
        try:
            await websocket.close(code=1011, reason=str(exc)[:120])
        except RuntimeError:
            pass
    except LiveRadarUnavailable as exc:
        logger.warning("Live Radar unavailable: %s", exc)
        try:
            await websocket.close(code=1013, reason=str(exc)[:120])
        except RuntimeError:
            pass
    except (RuntimeError, WebSocketDisconnect):
        return

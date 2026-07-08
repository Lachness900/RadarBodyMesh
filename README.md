# mmYoga: Privacy-Preserving mmWave Pose Matching

mmYoga is a COMP6733 IoT project for displaying and eventually classifying body poses from mmWave radar point clouds. The current repository contains the TI radar driver, the team's original offline `.dat` visualizer, and a web dashboard MVP that can stream replay data through a FastAPI backend.

The current MVP is replay-first: it uses recorded synchronized camera/radar `.dat` files to drive the dashboard before the final yoga pose model and live radar bridge are connected.

## Current Implementation

Implemented now:

- FastAPI backend with REST and WebSocket endpoints.
- Streaming parser for the team's Zstandard-compressed `.dat` recording format.
- Replay mode for real radar point-cloud data.
- Mock mode for frontend/backend smoke tests when replay data is unavailable.
- React + Vite dashboard with:
  - current pose panel,
  - pose confidence bars,
  - Three.js radar point-cloud view,
  - replay/mock input controls,
  - compact status strip.

Important limitation:

- Radar points in replay mode are real parsed `.dat` data.
- Pose predictions are still produced by a deterministic mock predictor.
- No final trained model is connected yet.

## Project Structure

```text
RadarBodyMesh/
  radar_driver/                 # TI mmWave ROS2 driver and messages
  point_visualizer/
    visualizer.py               # Original PyQt/OpenGL offline .dat visualizer
  mm_yoga/
    data/
      parser.py                 # Streaming .dat parser
      preprocess.py             # Replay filtering, feature shaping, JSON point conversion
    model/
      inference.py              # Predictor interface and mock predictor
    backend/
      app.py                    # FastAPI REST/WebSocket app
      replay.py                 # Replay stream -> dashboard message conversion
      schemas.py                # JSON message dataclasses
  frontend/
    src/                        # React dashboard source
    package.json                # Frontend scripts and dependencies
  OneDrive/                     # Local downloaded data, ignored by Git
  data/replay/                  # Uploaded replay files, ignored by Git
  models/                       # Local model artifacts, ignored by Git
  requirements.txt              # Python dependencies
```

## Local Data

The backend looks for a default replay file at:

```text
OneDrive/DepthCam_Radar_Cloud_Combined/cam_radar_1783260788477740516.dat
```

`OneDrive/` is intentionally ignored by Git because it contains large local data files. If the default replay file is missing, the backend and frontend can still run in mock mode.

Uploaded `.dat` files from the dashboard are stored under:

```text
data/replay/uploads/
```

That directory is also ignored by Git.

## Backend

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Start the backend on port `8000`:

```bash
uvicorn mm_yoga.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

If you use the local virtual environment:

```bash
.venv/bin/uvicorn mm_yoga.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Useful endpoints:

```text
GET  /health
GET  /api/sources
GET  /api/latest?source=replay
POST /api/replay-files?filename=<name>.dat
WS   /ws/predictions?source=replay&replay_file=<path>
```

Source modes:

- `mock`: generated point cloud and mock prediction.
- `replay`: parsed `.dat` radar replay and mock prediction.
- `auto`: replay if available, otherwise mock.

Environment variables:

```bash
MMYOGA_REPLAY_FILE=/path/to/file.dat
MMYOGA_PLAYBACK_SPEED=1.0
MMYOGA_MODEL_FILE=/path/to/model.file
```

`MMYOGA_MODEL_FILE` is reserved for a future model. At this stage, model loading is not implemented and predictions remain mock.

## Frontend

Install dependencies:

```bash
cd frontend
npm install
```

Start the Vite dev server:

```bash
npm run dev
```

Build check:

```bash
npm run build
```

The frontend defaults to:

```text
http://localhost:8000
ws://localhost:8000/ws/predictions
```

Override backend URLs if needed:

```bash
VITE_MMYOGA_API_URL=http://localhost:8000 \
VITE_MMYOGA_WS_URL=ws://localhost:8000/ws/predictions \
npm run dev
```

If the backend WebSocket is unavailable, the frontend falls back to local mock data so the UI can still be reviewed.

## Dashboard Message Shape

The frontend consumes JSON only. It does not parse `.dat` files directly.

Example WebSocket message:

```json
{
  "timestamp_ms": 12345,
  "source": "replay",
  "prediction": {
    "label": "straight_pose",
    "confidence": 0.91,
    "probabilities": {
      "t_pose": 0.03,
      "straight_pose": 0.91,
      "warrior_pose": 0.04,
      "other_pose": 0.02
    }
  },
  "points": [
    {
      "x": 2.4,
      "y": 0.1,
      "z": 0.8,
      "intensity": 0,
      "velocity": 0
    }
  ],
  "point_sets": {
    "projected_radar": [],
    "filtered_radar": [],
    "raw_radar": []
  },
  "metrics": {
    "fps": 10,
    "latency_ms": 4.2
  }
}
```

The dashboard currently supports three radar display modes:

- `Projected`: filtered radar history with `x` flattened, matching the current visualizer-style radar panel convention.
- `Filtered`: filtered radar points with xyz preserved.
- `Raw`: raw radar points from the replay file.

## Pose Labels

Current labels:

```text
t_pose
straight_pose
warrior_pose
other_pose
```

These match the current collection plan:

- T pose
- Straight pose
- Warrior pose
- Other or unrelated poses

## Original Python Visualizer

The team's existing offline visualizer remains at:

```text
point_visualizer/visualizer.py
```

Example usage:

```bash
python point_visualizer/visualizer.py \
  --file OneDrive/DepthCam_Radar_Cloud_Combined/cam_radar_1783260788477740516.dat
```

The web dashboard and the Python visualizer are separate display paths:

- Python visualizer: local GUI for inspecting recorded `.dat` files.
- Web dashboard: browser UI fed by backend JSON over WebSocket.

The MVP backend/frontend work does not modify the original visualizer.

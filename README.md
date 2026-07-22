# mmYoga: Privacy-Preserving mmWave Pose Matching

mmYoga is a COMP6733 IoT project for displaying and eventually classifying body poses from mmWave radar point clouds. The current repository contains the TI radar driver, the team's original offline `.dat` visualizer, and a web dashboard MVP that can stream replay data through a FastAPI backend.

The current MVP is replay-first: it uses recorded synchronized camera/radar `.dat` files to drive the dashboard before the final yoga pose model and live radar bridge are connected.

## Current Implementation

Implemented now:

- FastAPI backend with REST and WebSocket endpoints.
- Streaming parser for the team's Zstandard-compressed `.dat` recording format.
- Replay mode for real radar point-cloud data.
- Preliminary CNN pose classification from `pose_classifier.pt`.
- Mock mode for frontend/backend smoke tests when replay data is unavailable.
- React + Vite dashboard with:
  - current pose panel,
  - pose confidence bars,
  - Three.js radar point-cloud view,
  - replay/mock input controls,
  - compact status strip.

Important limitations:

- Radar points in replay mode are real parsed `.dat` data.
- Replay predictions use the preliminary trained model; mock mode remains synthetic.
- Live radar input is not connected yet.
- The checkpoint still needs evaluation on outlier recordings such as `t_pose_1_1`.

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
      inference.py              # CNN checkpoint loader and mock predictor
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

The current recording format is documented in `OneDrive/DepthCam_Radar_Cloud_Combined/FORMAT.md`. The backend parser supports that size-prefixed packet format and the older no-size packet format used by the first sample replay file, so both teammate-provided datasets and the existing smoke-test replay can be opened.

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
- `replay`: parsed `.dat` radar replay and CNN prediction.
- `auto`: replay if available, otherwise mock.

Environment variables:

```bash
MMYOGA_REPLAY_FILE=/path/to/file.dat
MMYOGA_PLAYBACK_SPEED=1.0
MMYOGA_MODEL_FILE=/path/to/pose_classifier.pt
```

The backend uses `pose_classifier.pt` by default. If the configured checkpoint
does not exist, it falls back to the deterministic mock predictor.

### Replay Model Preprocessing

Replay inference mirrors `point_visualizer/visualizer_with_classifier.py`: radar
points use the same subject ROI, each frame is centered at the origin, and the
classifier runs after at least 100 real points have accumulated. No zero-padding
is added to CNN input. Display points are processed separately, so positioning
the browser point cloud does not change the data received by the model.

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
    "label": "standing_pose",
    "confidence": 0.89,
    "probabilities": {
      "t_pose": 0.03,
      "standing_pose": 0.89,
      "warrior_1_pose": 0.02,
      "warrior_2_pose": 0.02,
      "angle_pose": 0.02,
      "other": 0.02
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

For replay messages, `metrics.fps` is the radar frame rate calculated from a
rolling window of recorded timestamps. `metrics.latency_ms` is the most recent
CNN inference time measured in the backend; it is not network round-trip
latency. The point count shown in the status strip is the number of points in
the currently selected display mode.

## Pose Labels

Current dashboard/model labels use the six classes stored in the checkpoint:

```text
t_pose
standing_pose
warrior_1_pose
warrior_2_pose
angle_pose
other
```

Use these exact snake_case keys for dataset metadata, backend predictions, and
future model outputs. They match the current collection plan:

- T pose
- Standing pose
- Right warrior pose
- Left warrior pose
- Angle pose
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

### Training Visualiser

We found it was efficient to use traditional image recognition techniques so the visualiser
was used to train the ai.

The training visualiser pipes the output of the file into the OBS virtual camera.

```text
point_visualizer/visualizer_training.py
```

To run, you must have OBS downloaded and have run the virtual camera atleast once so your
computer recognizes it. Similarly to the visualizer, it takes a file as argument.

This is purely for training the ai model and shouldn't be used for visualiasation purposes.

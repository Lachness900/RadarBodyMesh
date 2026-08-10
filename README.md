# mmPose: Privacy-Preserving mmWave Pose Matching

mmPose is a COMP6733 IoT project for displaying and classifying body poses from mmWave radar point clouds. The repository contains the TI radar driver, the team's offline `.dat` visualizer, and a web dashboard for Mock, recorded Replay, and Live Radar input.

Replay remains useful for repeatable demos. Live mode receives the point cloud forwarded by the ROS2 radar driver over local UDP and sends it through the same model pipeline.

## Current Implementation

Implemented now:

- FastAPI backend with REST and WebSocket endpoints.
- Streaming parser for the team's Zstandard-compressed `.dat` recording format.
- Replay mode for real radar point-cloud data.
- Live mode using the ROS2 point-cloud UDP bridge.
- PointNet-style pose classification directly from variable-length XYZ samples.
- Mock mode for frontend/backend smoke tests when replay data is unavailable.
- React + Vite dashboard with:
  - current pose panel,
  - pose confidence bars,
  - interactive SMPL pose-reference viewer,
  - Three.js radar point-cloud view,
  - mock/replay/live input controls,
  - 120/240 checkpoint selector for Replay and Live,
  - compact status strip.

Important limitations:

- Radar points in replay mode are real parsed `.dat` data.
- Replay predictions use the preliminary trained model; mock mode remains synthetic.
- Live mode requires the ROS2 radar driver and UDP bridge to be running.
- The 120- and 240-epoch candidate checkpoints still need live comparison before
  the team selects one for the final demonstration.
- The SMPL viewer includes the four recognized pose references: Standing,
  T Pose, Squat, and Angle.

### Point-cloud Model

The current model is a simplified PointNet-style classifier rather than the
older raster CNN. It consumes each centred, variable-length `N x 3` point set
directly. Inside the model, a shared `1 x 1` MLP maps each point from 3 to 256
features, masked max pooling produces one shape vector, and four auxiliary
statistics (`raw_scale`, `std_x`, `std_y`, and `skew_z`) are concatenated before
the classifier head. The backend then applies softmax and sends the same four
label probabilities expected by the existing dashboard.

The tracked candidates are:

```text
pointcloud_classifier120f.pt
pointcloud_classifier240.pt
```

Both use `angle_pose`, `squat`, `standing_pose`, and `t_pose`. The backend
prefers the 240-epoch candidate by default, but this is a runtime default rather
than a claim that it performs better on live radar.

## Project Structure

```text
RadarBodyMesh/
  radar_driver/                 # TI mmWave ROS2 driver and messages
  point_visualizer/
    visualizer.py               # Original PyQt/OpenGL offline .dat visualizer
    decoder.py                  # .dat and live UDP point-cloud readers
  mm_yoga/
    data/
      parser.py                 # Streaming .dat parser
      preprocess.py             # Shared filtering, feature shaping, JSON point conversion
    model/
      inference.py              # PointNet/raster checkpoint loaders and mock predictor
    backend/
      app.py                    # FastAPI REST/WebSocket app
      live.py                   # Shared live UDP receiver service
      replay.py                 # Recorded .dat source and replay timing
      schemas.py                # JSON message dataclasses
      stream.py                 # Shared replay/live preprocessing and inference
  frontend/
    src/                        # React dashboard source
    package.json                # Frontend scripts and dependencies
  tools/
    smpl_reference/             # Offline SMPL pose-reference GLB exporter
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

## SMPL Reference Pose

The four recognized pose-reference GLBs are tracked in Git and work immediately
after cloning or pulling the repository. They are static demonstrations, not
body meshes reconstructed from radar data.

To regenerate them, place the trusted source SMPL file under
`OneDrive/SMPL_model/`, install the Python requirements, and run:

```bash
.venv/bin/python tools/smpl_reference/generate_reference_pose.py \
  --model male \
  --all \
  --output-dir frontend/public/reference-poses
```

The T Pose is the original SMPL template. The other poses use the same template
with reviewed joint rotations. The source `.pkl` files stay local and ignored
by Git. The viewer supports rotation, zoom, pan, and view reset.

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
WS   /ws/predictions?source=replay&replay_file=<path>&model=pointcloud_240
WS   /ws/predictions?source=live&model=pointcloud_240
```

Source modes:

- `mock`: generated point cloud and mock prediction.
- `replay`: parsed `.dat` radar replay and model prediction.
- `live`: UDP radar frames and model prediction.
- `auto`: replay if available, otherwise mock.

Environment variables:

```bash
MMPOSE_REPLAY_FILE=/path/to/file.dat
MMPOSE_PLAYBACK_SPEED=1.0
MMPOSE_MODEL_FILE=/path/to/pointcloud_classifier240.pt
MMPOSE_RADAR_HOST=239.255.0.1
MMPOSE_RADAR_PORT=4200
MMPOSE_RADAR_INTERFACE=127.0.0.1
MMPOSE_RADAR_TIMEOUT=5.0
MMPOSE_PREDICTION_INTERVAL_MS=0
```

The dashboard can switch between the tracked 120- and 240-epoch checkpoints in
Replay or Live mode. Switching opens a new WebSocket; Replay restarts from the
beginning, while Live clears points accumulated for the previous model. The
backend accepts only model IDs returned by `GET /api/sources`, never arbitrary
checkpoint paths supplied by the browser.

The backend uses `pointcloud_classifier240.pt` by default, then falls back to
the 120-epoch candidate or the legacy `pose_classifier.pt` when the preferred
file is absent. Replay and Live require a valid configured checkpoint; Mock mode
uses its deterministic mock predictor. Existing `MMYOGA_*` variables remain
accepted as compatibility aliases.

To test the 120-epoch candidate instead:

```bash
MMPOSE_MODEL_FILE=pointcloud_classifier120f.pt \
  .venv/bin/uvicorn mm_yoga.backend.app:app --host 0.0.0.0 --port 8000
```

`MMPOSE_PREDICTION_INTERVAL_MS` throttles how often a ready model batch runs.
The default is `0`, matching the visualizer: predict whenever at least 100 real
radar points have accumulated. A positive interval can delay a ready batch, but
the backend never sends a sample with fewer than 100 points to the trained model.

### Radar Model Preprocessing

Replay and Live inference mirror the new
`point_cloud_visualizer_with_classifier.py` logic from the `Point_Cloud_AI`
branch: radar points use the same `[-10, 10]` bounds, each frame is centred at
the origin, and the classifier runs after at least 100 real points have
accumulated. The complete final radar frame is retained, so one sample may
contain slightly more than 100 points. No rasterization, external scale
normalization, or zero-padding is applied during single-sample inference.
Display points are processed separately, so positioning the browser point cloud
does not change the data received by the model.

### Live Radar

The live data path is:

```text
TI radar -> ROS2 driver -> /ti_mmwave/radar_scan_pcl -> PointCloudUDPBridge
         -> UDP 239.255.0.1:4200 -> FastAPI -> WebSocket -> frontend
```

Build and source the ROS2 workspace, then start the launch file that includes
the UDP bridge:

```bash
ros2 launch ti_mmwave_rospkg 1843_Standard.launch.py
```

Start FastAPI normally and choose **Live Radar** in the dashboard. The button
shows a waiting state until UDP frames arrive and an explicit error if the
stream is unavailable.

The confirmed UDP packet format remains:

```text
"::" + uint32 timestamp_us + uint16 payload_size + int16 xyz millimetres + ";;"
```

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

The frontend automatically connects to port `8000` on the same computer that
served the page. For example, these page and backend addresses are paired
without manual configuration:

```text
Dashboard: http://192.168.1.20:5173
API:       http://192.168.1.20:8000
WebSocket: ws://192.168.1.20:8000/ws/predictions
```

Override backend URLs if needed:

```bash
VITE_MMPOSE_API_URL=http://localhost:8000 \
VITE_MMPOSE_WS_URL=ws://localhost:8000/ws/predictions \
npm run dev
```

The former `VITE_MMYOGA_*` names remain supported as compatibility aliases.

### Phone/Hotspot Demo

Connect the computer to the phone hotspot, then start the backend with
`--host 0.0.0.0` and run the frontend normally:

```bash
.venv/bin/uvicorn mm_yoga.backend.app:app --host 0.0.0.0 --port 8000

cd frontend
npm run dev
```

Vite prints both `Local` and `Network` addresses. Open the `Network` address,
such as `http://172.20.10.2:5173`, on the phone. The dashboard derives the API
and WebSocket host automatically, so no `VITE_*` variables or source edits are
required. The computer firewall and hotspot must allow local-device traffic.

Mock can fall back to local generated data when the backend is unavailable.
Replay and Live instead show a disconnected or unavailable state, so synthetic
points are never presented as real radar input.

## Dashboard Message Shape

The frontend consumes JSON only. It does not parse `.dat` files directly.

Example WebSocket message:

```json
{
  "timestamp_ms": 12345,
  "source": "replay",
  "model_id": "pointcloud_240",
  "prediction": {
    "label": "standing_pose",
    "confidence": 0.89,
    "probabilities": {
      "standing_pose": 0.85,
      "t_pose": 0.03,
      "squat": 0.07,
      "angle_pose": 0.05
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
- `Raw`: raw radar points from the replay file or live UDP stream.

For Replay and Live messages, `metrics.fps` is the radar frame rate calculated
from a rolling window of source timestamps. `metrics.latency_ms` is the most recent
model inference time measured in the backend; it is not network round-trip
latency. The point count shown in the status strip is the number of points in
the currently selected display mode.

## Pose Labels

Current dashboard/model labels use the four classes stored in the checkpoint:

```text
standing_pose
t_pose
squat
angle_pose
```

Use these exact snake_case keys for dataset metadata, backend predictions, and
future model outputs. They match the current collection plan:

- Standing pose
- T pose
- Squat pose
- Angle pose

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

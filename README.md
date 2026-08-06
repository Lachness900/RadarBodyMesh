# mmYoga: Privacy-Preserving mmWave Pose Matching

mmYoga is a COMP6733 IoT project for displaying and classifying body poses from mmWave radar point clouds. The repository contains the TI radar driver, the team's offline `.dat` visualizer, and a web dashboard for Mock, recorded Replay, and Live Radar input.

Replay remains useful for repeatable demos. Live mode receives the point cloud forwarded by the ROS2 radar driver over local UDP and sends it through the same model pipeline.

## Current Implementation

Implemented now:

- FastAPI backend with REST and WebSocket endpoints.
- Streaming parser for the team's Zstandard-compressed `.dat` recording format.
- Replay mode for real radar point-cloud data.
- Live mode using the ROS2 point-cloud UDP bridge.
- Preliminary CNN pose classification from `pose_classifier.pt`.
- Mock mode for frontend/backend smoke tests when replay data is unavailable.
- React + Vite dashboard with:
  - current pose panel,
  - pose confidence bars,
  - interactive SMPL pose-reference viewer,
  - Three.js radar point-cloud view,
  - mock/replay/live input controls,
  - compact status strip.

Important limitations:

- Radar points in replay mode are real parsed `.dat` data.
- Replay predictions use the preliminary trained model; mock mode remains synthetic.
- Live mode requires the ROS2 radar driver and UDP bridge to be running.
- The checkpoint still needs evaluation on outlier recordings such as `t_pose_1_1`.
- The SMPL viewer includes the five recognized pose references. `Other Pose`
  intentionally has no standard reference.

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
      inference.py              # CNN checkpoint loader and mock predictor
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

The five recognized pose-reference GLBs are tracked in Git and work immediately
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
by Git. The viewer supports rotation, zoom, pan, and view reset. `Other Pose`
has no reference because that class represents multiple unrelated poses.

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
WS   /ws/predictions?source=live
```

Source modes:

- `mock`: generated point cloud and mock prediction.
- `replay`: parsed `.dat` radar replay and CNN prediction.
- `live`: UDP radar frames and CNN prediction.
- `auto`: replay if available, otherwise mock.

Environment variables:

```bash
MMYOGA_REPLAY_FILE=/path/to/file.dat
MMYOGA_PLAYBACK_SPEED=1.0
MMYOGA_MODEL_FILE=/path/to/pose_classifier.pt
MMYOGA_RADAR_HOST=239.255.0.1
MMYOGA_RADAR_PORT=4200
MMYOGA_RADAR_INTERFACE=127.0.0.1
MMYOGA_RADAR_TIMEOUT=5.0
```

The backend uses `pose_classifier.pt` by default. Replay and Live require a
valid configured checkpoint; Mock mode uses its deterministic mock predictor.

### Radar Model Preprocessing

Replay and Live inference mirror `point_visualizer/visualizer_with_classifier.py`: radar
points use the same subject ROI, each frame is centered at the origin, and the
classifier runs after at least 100 real points have accumulated. No zero-padding
is added to CNN input. Display points are processed separately, so positioning
the browser point cloud does not change the data received by the model.

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
- `Raw`: raw radar points from the replay file or live UDP stream.

For Replay and Live messages, `metrics.fps` is the radar frame rate calculated
from a rolling window of source timestamps. `metrics.latency_ms` is the most recent
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
- Warrior Pose 1
- Warrior Pose 2
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

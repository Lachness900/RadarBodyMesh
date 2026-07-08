# mmYoga: Privacy-Preserving mmWave Yoga Pose Matching

mmYoga is a COMP6733 IoT project for visualizing and eventually classifying yoga/body poses from mmWave radar point clouds. The repository currently contains the TI radar driver, an offline Python point-cloud visualizer, recorded/local data support, and the first version of the web dashboard.

The current milestone focuses on the frontend display layer. The dashboard can run independently: when a backend WebSocket is available, it displays backend data; when the backend is unavailable, it falls back to local mock data so the UI can still be reviewed.

## Current Stage

Implemented in this stage:

- `frontend/`: React + Vite dashboard.
- `frontend/src/hooks/usePredictionStream.js`: owns the WebSocket connection and falls back to a local mock stream when the backend is not available.
- `frontend/src/components/RadarPointCloud.jsx`: renders the radar point cloud in 3D with Three.js.
- `frontend/src/components/PredictionPanel.jsx`: shows the current predicted pose.
- `frontend/src/components/ConfidenceBars.jsx`: shows confidence values for each pose class.
- `frontend/src/components/StatusStrip.jsx`: shows connection status, source, FPS, latency, and point count.
- `point_visualizer/visualizer.py`: the existing PyQt/PyOpenGL offline visualizer from the team. This has not been modified by the frontend work.

This stage does not merge model training, final inference, or visualizer refactoring into the main flow. The frontend prediction may still be mock data. The point-cloud view only displays the `points` received from the backend or mock stream.

## Frontend Data Flow

By default, the frontend connects to:

```text
ws://localhost:8000/ws/predictions
```

The URL can be overridden with:

```bash
VITE_MMYOGA_WS_URL=ws://localhost:8000/ws/predictions npm run dev
```

The dashboard expects WebSocket messages in this shape:

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
      "intensity": 14.2,
      "velocity": 0
    }
  ],
  "metrics": {
    "fps": 10,
    "latency_ms": 42
  }
}
```

If the WebSocket connection fails, `usePredictionStream` generates local mock messages with the same structure, so the page can still run without the backend.

## Radar Point Cloud Display

`RadarPointCloud.jsx` uses Three.js `BufferGeometry` to render the point cloud:

- Point positions use the original `x`, `y`, and `z` values.
- The frontend does not perform 2D projection or flatten the point cloud.
- The `z` axis is treated as the vertical height axis, matching the existing Python visualizer convention as closely as possible.
- Point color is lightly mapped from `intensity` for readability. This only affects display and does not modify point coordinates.
- The camera auto-frames incoming data at first. After the user manually rotates or zooms, the camera preserves the user's chosen viewpoint.

In short, the 3D panel is meant to display the radar frame as directly as possible, not to preprocess data for the model.

## Run Frontend

Enter the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the local URL printed by Vite.

Build check:

```bash
npm run build
```

The build may show a Vite chunk-size warning because Three.js is large. This is a warning, not a build failure.

## Python Visualizer

The existing offline visualizer remains here:

```text
point_visualizer/visualizer.py
```

The visualizer and frontend are two separate display paths:

- Python visualizer: local GUI for inspecting recorded `.dat` radar/camera files.
- Web frontend: browser dashboard for displaying pose predictions and radar points from WebSocket JSON messages.

The current frontend work does not modify `point_visualizer/visualizer.py`.

## Current Pose Labels

The dashboard currently uses four pose labels:

- `t_pose`
- `straight_pose`
- `warrior_pose`
- `other_pose`

These match the current data collection plan: T pose, straight pose, warrior pose, and unrelated/other poses.

## Important Notes

- The frontend can run independently without the backend.
- Before a real replay backend is connected, prediction and points may come from the local mock stream.
- The frontend consumes JSON messages only. It does not parse `.dat` files directly.
- The AI model is not connected in this stage. Future backend work can replace mock prediction output with real model inference.


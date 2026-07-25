# SMPL Reference Pose Export

This tool is the first, deliberately small stage of the mmYoga SMPL reference
work. It exports the canonical SMPL template as a static T-pose GLB and creates
a front/side PNG preview for team approval.

It does **not** reconstruct a person from radar data. It also does not yet apply
SMPL joint rotations, body shape parameters, or pose blend shapes.

## Input

By default the tool reads one of these local, Git-ignored files:

```text
OneDrive/SMPL_model/smpl_m.pkl
OneDrive/SMPL_model/smpl_f.pkl
```

Python pickle files can execute code while loading. Only use the trusted SMPL
files supplied to this project. The loader restricts globals to the NumPy and
SciPy sparse types present in the inspected files.

## Generate the T Pose

Use the project virtual environment so NumPy and Matplotlib versions remain
consistent:

```bash
.venv/bin/python tools/smpl_reference/generate_reference_pose.py --model male
```

To inspect the female template instead:

```bash
.venv/bin/python tools/smpl_reference/generate_reference_pose.py --model female
```

Generated files are local and ignored by Git:

```text
data/smpl_reference/t_pose_male.glb
data/smpl_reference/t_pose_male.png
data/smpl_reference/t_pose_male.json
```

The GLB preserves the model's native metre-scale coordinates. The PNG shows
front and side views. The JSON report records geometry bounds and file hashes.

To make the approved local T Pose available to the Vite dashboard:

```bash
.venv/bin/python tools/smpl_reference/generate_reference_pose.py \
  --model male \
  --output frontend/public/reference-poses/t_pose_male.glb
```

The generated GLB remains ignored by Git until its redistribution requirements
have been confirmed. Its preview and JSON report remain in
`data/smpl_reference/` rather than being served by Vite.

## Stage-One Boundary

Before connecting anything to the dashboard:

1. Confirm that the preview is the expected canonical T Pose.
2. Confirm which body template should be used as the shared reference.
3. Confirm the licence and attribution requirements for generated GLB assets.

Later pose presets should use a maintained SMPL implementation for linear blend
skinning rather than extending this exporter with hand-written pose math.

# SMPL Reference Pose Export

This tool generates the static reference poses shown in the mmYoga dashboard.
It does not reconstruct a person from radar data.

The T Pose uses the original SMPL template directly. The other poses start from
the same template, apply the joint rotations in `pose_presets.py`, and use the
official `smplx` linear blend skinning implementation to create the posed mesh.

## Source Model

Place a trusted local SMPL file in:

```text
OneDrive/SMPL_model/smpl_m.pkl
```

The source `.pkl` stays local and is ignored by Git.

## Generate

Install the project requirements, then generate all frontend assets:

```bash
.venv/bin/python tools/smpl_reference/generate_reference_pose.py \
  --model male \
  --all \
  --output-dir frontend/public/reference-poses
```

The dashboard includes references for:

- T Pose
- Standing Pose
- Warrior Pose 1
- Warrior Pose 2
- Angle Pose

`Other Pose` has no single reference because it represents many unrelated
movements.

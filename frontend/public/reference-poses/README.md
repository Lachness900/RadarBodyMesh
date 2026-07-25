# Local SMPL Reference Assets

Generated SMPL `.glb` files in this directory are ignored by Git until the
project confirms their redistribution and attribution requirements.

Generate the stage-one T Pose used by the dashboard:

```bash
.venv/bin/python tools/smpl_reference/generate_reference_pose.py \
  --model male \
  --output frontend/public/reference-poses/t_pose_male.glb
```

The frontend loads this file as `/reference-poses/t_pose_male.glb`. When the
file is missing, the Current Pose panel reports that the reference is
unavailable without affecting pose prediction or radar rendering. Preview and
report files remain under `data/smpl_reference/`.

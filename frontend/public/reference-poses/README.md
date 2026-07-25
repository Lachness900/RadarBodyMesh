# Local SMPL Reference Assets

`t_pose_male.glb` is tracked in Git and is available immediately after cloning
or pulling the repository. Other generated SMPL `.glb` files remain ignored
until the project confirms their redistribution requirements.

Optional command for intentionally regenerating the T Pose:

```bash
.venv/bin/python tools/smpl_reference/generate_reference_pose.py \
  --model male \
  --output frontend/public/reference-poses/t_pose_male.glb
```

The frontend resolves the file relative to Vite's configured base URL. If it is
missing, the Current Pose panel reports that the reference is unavailable
without affecting pose prediction or radar rendering. Preview and report files
remain under `data/smpl_reference/`.

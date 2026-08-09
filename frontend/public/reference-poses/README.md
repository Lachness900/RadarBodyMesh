# SMPL Reference Assets

The four `.glb` files in this directory are tracked frontend assets. They are
static reference demonstrations and are not generated from live radar data.

Current references: Standing, T Pose, Squat, and Angle.

To regenerate them from the trusted local SMPL source:

```bash
.venv/bin/python tools/smpl_reference/generate_reference_pose.py \
  --model male \
  --all \
  --output-dir frontend/public/reference-poses
```

The frontend loads the file matching the predicted pose label.

# Camera calibration records

One YAML file per camera, named after its role: `wrist.yaml`, `front.yaml`.
These files are the single source of truth for camera geometry — the real
capture layer (`so101.real.cameras`) and the Isaac Lab scene
(`so101.scenes.tabletop`) both read them, so the simulated and physical
cameras cannot drift apart.

Produced by `scripts/calibrate_intrinsics.py` and
`scripts/calibrate_handeye.py`. Do not hand-edit; regenerate instead.

## Why `K_virtual` exists

Isaac Lab renders an ideal pinhole only:

- `Camera._update_intrinsic_matrices` hardcodes `f_y = f_x`, `c_x = W/2`, `c_y = H/2`
- `spawn_camera` discards aperture offsets entirely (NVIDIA ticket OM-42611)
- `PinholeCameraCfg.from_intrinsic_matrix` silently drops `c_x`/`c_y` and averages `f_x`/`f_y`

So rather than bending the renderer, real frames are remapped onto an ideal
pinhole — `K_virtual` — that Isaac reproduces exactly. Lens distortion is
removed in the same `cv2.remap` call. `K_virtual` is required to have
`fx == fy` and to be centred; the loader rejects anything else, because a
`K_virtual` Isaac cannot render would show up much later as an unexplained
gate failure.

## Fields

| field | meaning |
|---|---|
| `name` | camera role: `wrist` or `front` |
| `device` | V4L2 node, e.g. `/dev/video0` |
| `resolution` | `[width, height]` — must match the capture resolution exactly |
| `fps` | capture rate |
| `K` | measured intrinsic matrix (3x3, row-major) |
| `dist` | measured OpenCV distortion coefficients |
| `K_virtual` | rectification target; `fx == fy`, principal point at the centre |
| `alpha` | `getOptimalNewCameraMatrix` alpha used (0 = crop to all-valid pixels) |
| `extrinsic.parent` | `base` for the front camera, the gripper link for the wrist |
| `extrinsic.convention` | always `ros` (OpenCV: +Z optical axis, +Y down) |
| `extrinsic.pos` / `quat_wxyz` | camera pose relative to `parent` |
| `exposure` | the V4L2 controls that were locked, and any the device does not expose |
| `calib_rms_px` | OpenCV RMS reprojection error from intrinsic calibration (gate: < 0.3) |
| `handeye_rmse_px` | reprojection RMSE of the hand-eye solution |
| `hfov_loss_frac` | horizontal FOV lost to `alpha=0` cropping (> 0.10 triggers review) |
| `date` | when the calibration was taken |

## Gotchas

- **Calibrate at the resolution you run at.** USB cameras crop or bin
  differently per mode, so a `K` measured at 1280x720 and halved is quietly
  wrong. The capture layer refuses a calibration whose resolution does not
  match.
- **`convention` is always `ros`.** That is what `cv2.solvePnP` and
  `cv2.calibrateHandEye` return and what Isaac Lab's `OffsetCfg` means by
  `convention="ros"`, so the numbers transfer with no conversion layer to get
  a sign wrong in.
- The Orbbec Gemini exposes no white balance control over V4L2. Its
  `exposure.unsupported_controls` records that rather than leaving it as a
  silent assumption.

See `example.yaml` for the shape. It is not loaded by anything — the loader
looks for `wrist.yaml` / `front.yaml`.

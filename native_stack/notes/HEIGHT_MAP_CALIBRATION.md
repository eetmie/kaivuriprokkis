# Height map calibration (2026-11-20, `flat-calib-001`)

Goal: a height map good enough for the VLA to see sand pile height (cm, not mm). Fallback: a level-corrected
cabin view (pixels coloured by height above the ground), which needs only the cabin camera's tilt and height.

## Session

`pikkumasi2tb:~/data_collection/native_sessions/flat-calib-001` (copy on joel-omen `~/Desktop/flat-calib-001`,
Rerun files of every episode in `~/Desktop/flat-calib-001-rerun/`). Recorded from this branch's build:

```bash
cd ~/kp-depth-close/native_stack
python3 tools/record_native.py --session ~/data_collection/native_sessions/flat-calib-001 [--resume] \
  --task "flat plane calibration" --resolution 640x480 --depth both --enable-slew \
  --depth-units-um 100 --cam1-disparity-shift 25 --cam2-disparity-shift 0 --max-episode-s 420 \
  --gamepad /dev/input/by-id/usb-Microsoft_Controller_3039373030313334303932333135-event-joystick \
  --serial /dev/serial/by-id/usb-RPi_IMU_Reader_4250305031363910-if00
```

| Ep | Take |
| --- | --- |
| 0, 1 | Flat white board, arm through many poses, bucket tip touches on the board (17 found) |
| 2 | Static, 31.5 mm brick in view |
| 3 | Tipping the machine front up (to 24°), tip on the board |
| 4 | Driving around a 42 mm cardboard box |
| 5 | Machine still, box spinning in front |

## Findings

- **FK:** touches repeat within ~1 mm; the touch plane is 0.5° off the FK base plane (board slope or a small boom/arm
  offset; not separated).
- **Tilt pivot:** the controller tilted the FK about the ground origin; the machine tips about the track end. With
  `robot.level_frame.tilt_pivot_dist: 0.16` (front/back; `null` = old behaviour) the tip height error on ep 3 drops
  from p90 31 mm to 3.5 mm (0.14 and 0.18 m are worse). Side tipping is not modelled.
- **Depth:** about 2 mm flat residual per frame; fill only 45–63 % on the white board without the emitter (textured
  sand fills nearly fully). The stereo dome seen on `sand-depth-001` (stock settings) is gone with the close-range
  settings: the fitted correction changes planarity by 0.1 mm, so `fit_mounts.py` leaves it out.
- **Old `sand_height.py calibrate`** (a rough fit for view-correction tests): the cabin camera mount comes out 4.5–8.7 m
  off (flat ground cannot see its height apart from the ground offset, and the fit picks a degenerate split), and the
  overlap step fails on a moving object. To be removed once `fit_mounts.py` has replaced it on a rough-sand take.

## `tools/fit_mounts.py`

Ground plane from the tip touches (so heights share the FK frame), then on the flat take: the cabin camera's two
tilts and height, and the wrist camera's mount without its offset along the arm hinge axis. The cabin camera's yaw
and x/y, and the wrist camera's hinge-axis offset, only move points sideways, which flat ground cannot see: they come
from `--start` (the old calibration) or the tape (`--cabin-xy`, `--wrist-y`). Writes the `sand_height.py` calibration
format, so `sand_height.py view --calibration` and the ROS bridge read it.

```bash
python3 tools/fit_mounts.py ~/data_collection/native_sessions/flat-calib-001 --flat 0 --holdout 1 \
  --object 2:31.5 --object 4:42 --object 5:42 --start ~/depth-live-bench/sand-calibration.json -o mounts.json
```

First result (ground: 17 touches, residual p50/p90 0.8/3.0 mm):

| | Flat held out (ep 1) p50/p90 | Box 42 mm (ep 4 / ep 5) | Brick 31.5 mm (ep 2) |
| --- | --- | --- | --- |
| Wrist camera | 2.5 / 5.7 mm | 43.7 / 43.5 mm | 25.0 mm |
| Cabin camera | 2.8 / 9.0 mm | 46.9 / 47.5 mm | not found |

Good enough for cm-scale piles; the cabin camera still reads the box ~5 mm high (its yaw and x/y are the old rough
values) and the brick reads low in the wrist camera (check its footprint in Rerun).

## Next week

1. Tape-measure the cabin camera from the slew axis (x forward, y left, on the slew link) and the wrist camera's
   sideways offset from the arm centre line, to ~1 cm.
2. Record about 5 min on rough sand with 2–3 mounds (same command with `--resume` or a new session): flat-ish
   ground with tip touches, then the arm still while slewing slowly across the mounds, then a few still holds where
   both cameras see one mound.
3. Rerun `fit_mounts.py` with `--cabin-xy`/`--wrist-y`; if the box/mound heights still disagree between the cameras,
   add a yaw/xy refinement on the slewing stretch (heights of a static mound must not change with slew).
4. Deploy `tilt_pivot_dist` (rebuild `kaivuri-control` in `~/kp-depth-close`, check with a tip touch while tipped).
5. Then the ROS side: `kaivuri_bridge.py` publishes depth + `camera_info` and reads `mounts.json`; FastDEM; compare
   its map with these numbers on a replay. Remove `sand_height.py calibrate`.

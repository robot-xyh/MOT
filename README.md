# Infrared Point-Target MOT

Utilities for clipping the last segment of an infrared video and tracking small point targets.

## Install

```bash
python3 -m pip install -r requirements.txt
```

If dependencies are installed into a local `.deps` directory, run commands with:

```bash
PYTHONPATH=.deps python3 ...
```

## Manual Tracking

Reuse saved initialization boxes:

```bash
PYTHONPATH=.deps python3 scripts/track_ir_points.py \
  --input a.mp4 \
  --init-boxes outputs/init_boxes.json \
  --vis-out outputs/a_last120_v2_manual_tracks.mp4 \
  --csv-out outputs/manual_v2_tracks.csv \
  --max-misses 60
```

Select targets manually:

```bash
PYTHONPATH=.deps python3 scripts/track_ir_points.py \
  --input a.mp4 \
  --manual-init \
  --init-frame 0 \
  --save-init-boxes outputs/init_boxes.json
```

In the OpenCV selection window, draw one box per target, press `Enter` or `Space` to confirm each box, and press `Esc` when all targets are selected.

## Auto-Initialized Tracking

For unattended systems, use auto initialization. The first few frames are used to discover stable point candidates, then tracking is locked onto those candidates:

```bash
PYTHONPATH=.deps python3 scripts/track_ir_points.py \
  --input a.mp4 \
  --auto-init \
  --vis-out outputs/a_last120_auto_init_tracks.mp4 \
  --csv-out outputs/auto_init_tracks.csv \
  --save-auto-init outputs/auto_init.json
```

Useful controls:

```bash
--auto-init-seconds 2.0
--auto-init-min-hits 5
--auto-init-min-hit-ratio 0.25
--auto-init-max-targets 8
--auto-hold-lost-frames 60
```

The built-in `classical` detector is a non-semantic infrared point-candidate detector. It is useful as a fallback or for data mining, but it can select stable background hot points. For an operational drone system, prefer feeding detections from a recognition model through the external detector interface below.

## Free Automatic Tracking

```bash
PYTHONPATH=.deps python3 scripts/track_ir_points.py \
  --input a.mp4 \
  --vis-out outputs/a_last120_tracks.mp4 \
  --csv-out outputs/tracks.csv
```

## External Detector Input

External detector CSV format:

```text
frame_idx,x,y,width,height,score,class_name
```

Run with:

```bash
PYTHONPATH=.deps python3 scripts/track_ir_points.py \
  --input a.mp4 \
  --auto-init \
  --detector external-csv \
  --detections-in outputs/detections.csv \
  --external-high-score 0.5 \
  --external-low-score 0.1
```

Generated videos, CSV files, and local input videos are intentionally ignored by git.

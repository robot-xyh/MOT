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

## Automatic Tracking

```bash
PYTHONPATH=.deps python3 scripts/track_ir_points.py \
  --input a.mp4 \
  --vis-out outputs/a_last120_tracks.mp4 \
  --csv-out outputs/tracks.csv
```

Generated videos, CSV files, and local input videos are intentionally ignored by git.

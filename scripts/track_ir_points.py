#!/usr/bin/env python3
"""Clip the last segment of an IR video and track small bright point targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    area: int
    width: int
    height: int
    intensity: float
    score: float
    contrast: float = 0.0
    source: str = "global"


@dataclass(frozen=True)
class InitBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class TrackRecord:
    frame_idx: int
    time_s: float
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    intensity: float
    score: float
    confidence: float
    response: float
    source: str
    association_stage: str
    state: str


@dataclass
class Track:
    track_id: int
    state: np.ndarray
    cov: np.ndarray
    hits: int = 1
    age: int = 1
    misses: int = 0
    confirmed: bool = False
    last_detection: Detection | None = None
    confidence: float = 1.0
    response_ema: float = 0.0
    low_response_streak: int = 0
    last_association_stage: str = "init"
    response_history: deque[float] = field(default_factory=deque)
    records: list[TrackRecord] = field(default_factory=list)
    trail: deque[tuple[float, float]] = field(default_factory=deque)

    @property
    def position(self) -> tuple[float, float]:
        return float(self.state[0]), float(self.state[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.state[2]), float(self.state[3])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep the last N seconds of an infrared video, detect small bright "
            "point targets, and produce multi-object tracks."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input video path.")
    parser.add_argument(
        "--clip-out",
        type=Path,
        help="Output path for the clipped video. Defaults to outputs/<stem>_last120.mp4.",
    )
    parser.add_argument(
        "--vis-out",
        type=Path,
        help="Output path for the track visualization video.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("outputs/tracks.csv"),
        help="Output path for the track CSV.",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=120.0,
        help="Number of seconds to keep from the end of the video.",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Track --input directly and do not create a clipped video.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=7.0,
        help="Gaussian sigma used to estimate local background.",
    )
    parser.add_argument(
        "--threshold-k",
        type=float,
        default=6.0,
        help="Robust high-pass threshold multiplier: median + k * MAD.",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=8.0,
        help="Minimum high-pass threshold in gray levels.",
    )
    parser.add_argument("--min-area", type=int, default=1, help="Minimum blob area.")
    parser.add_argument("--max-area", type=int, default=12, help="Maximum blob area.")
    parser.add_argument("--max-width", type=int, default=5, help="Maximum blob width.")
    parser.add_argument("--max-height", type=int, default=5, help="Maximum blob height.")
    parser.add_argument(
        "--ignore-corner-size",
        type=int,
        default=56,
        help="Bottom-left and bottom-right corner size ignored as overlay regions.",
    )
    parser.add_argument(
        "--gate-distance",
        type=float,
        default=15.0,
        help="Maximum pixel distance for detection-to-track assignment.",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Consecutive or accumulated hits required before a track is confirmed.",
    )
    parser.add_argument(
        "--max-misses",
        type=int,
        default=10,
        help="Maximum consecutive missed frames before deleting a track.",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=5,
        help="Minimum hit count required to write a track to CSV.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=30,
        help="Number of recent points drawn for each confirmed track.",
    )
    parser.add_argument(
        "--process-noise",
        type=float,
        default=2.0,
        help="Kalman process noise strength.",
    )
    parser.add_argument(
        "--measurement-noise",
        type=float,
        default=2.0,
        help="Kalman measurement noise standard deviation in pixels.",
    )
    parser.add_argument(
        "--draw-detections",
        action="store_true",
        help="Draw raw detections as small gray crosses in the visualization.",
    )
    parser.add_argument(
        "--manual-init",
        action="store_true",
        help="Open a GUI window to manually draw initial boxes for the targets to track.",
    )
    parser.add_argument(
        "--init-frame",
        type=int,
        default=0,
        help="Frame index in the tracked video used for manual initialization.",
    )
    parser.add_argument(
        "--save-init-boxes",
        type=Path,
        help="Save manually selected boxes to this JSON file.",
    )
    parser.add_argument(
        "--init-boxes",
        type=Path,
        help="Load target initialization boxes from a JSON file.",
    )
    parser.add_argument(
        "--init-search-radius",
        type=float,
        default=8.0,
        help="Allowed distance from each selected box center to an initial detection.",
    )
    parser.add_argument(
        "--manual-search-radius",
        type=int,
        default=28,
        help="Local search radius around each manually initialized target.",
    )
    parser.add_argument(
        "--manual-min-score",
        type=float,
        default=1.5,
        help="Minimum local high-pass score accepted for manual target tracking.",
    )
    parser.add_argument(
        "--manual-global-correction",
        action="store_true",
        help="Use global high-confidence detections to correct manually initialized tracks.",
    )
    parser.add_argument(
        "--manual-correction-radius",
        type=float,
        default=8.0,
        help="Maximum distance for optional global correction in manual tracking mode.",
    )
    parser.add_argument(
        "--manual-drop-lost",
        action="store_true",
        help="Drop manually initialized tracks after --max-misses missed frames.",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=5.0,
        help=(
            "High-confidence candidate threshold for ByteTrack-style first-stage matching. "
            "The effective high threshold is also constrained by --high-percentile."
        ),
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=3.0,
        help="Low-confidence candidate threshold used only to recover existing tracks.",
    )
    parser.add_argument(
        "--high-percentile",
        type=float,
        default=90.0,
        help=(
            "Per-frame candidate score percentile used with --high-threshold. "
            "Set to 0 to disable percentile adaptation."
        ),
    )
    parser.add_argument(
        "--tbd-window",
        type=int,
        default=5,
        help="Temporal response window for manual track-before-detect smoothing.",
    )
    parser.add_argument(
        "--tbd-min-response",
        type=float,
        default=1.5,
        help="Minimum smoothed local response for manual TBD detections.",
    )
    parser.add_argument(
        "--association",
        choices=("standard", "bytetrack"),
        default="bytetrack",
        help="Association strategy for automatic tracking.",
    )
    parser.add_argument(
        "--motion-comp",
        choices=("none", "phase"),
        default="none",
        help="Optional global translation compensation between adjacent frames.",
    )
    parser.add_argument(
        "--debug-detections",
        type=Path,
        help="Optional CSV path for per-frame candidate detections.",
    )
    return parser.parse_args()


def default_clip_path(input_path: Path, clip_seconds: float) -> Path:
    seconds = int(round(clip_seconds))
    return Path("outputs") / f"{input_path.stem}_last{seconds}.mp4"


def default_vis_path(clip_path: Path) -> Path:
    return clip_path.with_name(f"{clip_path.stem}_tracks.mp4")


def run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        joined = " ".join(command)
        raise RuntimeError(f"Command failed with exit code {exc.returncode}: {joined}") from exc


def get_video_duration(video_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required but was not found in PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed for {video_path}: {exc.stderr}") from exc
    data = json.loads(result.stdout)
    try:
        return float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not read video duration from {video_path}") from exc


def clip_last_seconds(input_path: Path, output_path: Path, clip_seconds: float) -> tuple[float, float]:
    duration = get_video_duration(input_path)
    start_s = max(0.0, duration - clip_seconds)
    keep_s = min(clip_seconds, duration)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{keep_s:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    run_command(command)
    return start_s, keep_s


def build_ignore_mask(height: int, width: int, corner_size: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if corner_size <= 0:
        return mask
    size = min(corner_size, height, width)
    mask[height - size : height, 0:size] = 255
    mask[height - size : height, width - size : width] = 255
    return mask


def robust_threshold(values: np.ndarray, threshold_k: float, min_threshold: float) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values.astype(np.float32) - median)))
    sigma = 1.4826 * mad
    return max(min_threshold, median + threshold_k * sigma)


def compute_highpass(frame: np.ndarray, ignore_mask: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    highpass = cv2.subtract(gray, background)
    highpass[ignore_mask > 0] = 0
    return gray, highpass


def detect_points(
    frame: np.ndarray,
    ignore_mask: np.ndarray,
    sigma: float,
    threshold_k: float,
    min_threshold: float,
    min_area: int,
    max_area: int,
    max_width: int,
    max_height: int,
) -> list[Detection]:
    gray, highpass = compute_highpass(frame, ignore_mask, sigma)
    threshold = robust_threshold(highpass, threshold_k, min_threshold)
    return detect_candidates_from_highpass(
        gray=gray,
        highpass=highpass,
        ignore_mask=ignore_mask,
        threshold=threshold,
        min_area=min_area,
        max_area=max_area,
        max_width=max_width,
        max_height=max_height,
        source="global",
    )


def detect_candidates_from_highpass(
    gray: np.ndarray,
    highpass: np.ndarray,
    ignore_mask: np.ndarray,
    threshold: float,
    min_area: int,
    max_area: int,
    max_width: int,
    max_height: int,
    source: str,
) -> list[Detection]:
    _, binary = cv2.threshold(highpass, threshold, 255, cv2.THRESH_BINARY)
    binary[ignore_mask > 0] = 0

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    detections: list[Detection] = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area or area > max_area:
            continue
        if w > max_width or h > max_height:
            continue

        component_mask = labels[y : y + h, x : x + w] == label
        component_hp = highpass[y : y + h, x : x + w][component_mask]
        component_gray = gray[y : y + h, x : x + w][component_mask]
        if component_hp.size == 0:
            continue

        cx, cy = centroids[label]
        score = float(np.max(component_hp))
        contrast = float(np.mean(component_hp))
        detections.append(
            Detection(
                x=float(cx),
                y=float(cy),
                area=int(area),
                width=int(w),
                height=int(h),
                intensity=float(np.max(component_gray)),
                score=score,
                contrast=contrast,
                source=source,
            )
        )
    return detections


def detect_tiered_points(
    frame: np.ndarray,
    ignore_mask: np.ndarray,
    sigma: float,
    high_threshold: float,
    low_threshold: float,
    high_percentile: float,
    min_area: int,
    max_area: int,
    max_width: int,
    max_height: int,
) -> tuple[list[Detection], list[Detection], list[Detection]]:
    gray, highpass = compute_highpass(frame, ignore_mask, sigma)
    low_candidates = detect_candidates_from_highpass(
        gray=gray,
        highpass=highpass,
        ignore_mask=ignore_mask,
        threshold=low_threshold,
        min_area=min_area,
        max_area=max_area,
        max_width=max_width,
        max_height=max_height,
        source="global_low",
    )
    effective_high_threshold = high_threshold
    if low_candidates and high_percentile > 0:
        percentile = min(max(high_percentile, 0.0), 100.0)
        scores = np.array([d.score for d in low_candidates], dtype=np.float32)
        effective_high_threshold = max(high_threshold, float(np.percentile(scores, percentile)))
    high_candidates = [
        Detection(
            x=d.x,
            y=d.y,
            area=d.area,
            width=d.width,
            height=d.height,
            intensity=d.intensity,
            score=d.score,
            contrast=d.contrast,
            source="global_high",
        )
        for d in low_candidates
        if d.score >= effective_high_threshold
    ]
    high_keys = {(round(d.x, 2), round(d.y, 2)) for d in high_candidates}
    recovery_candidates = [
        d for d in low_candidates if (round(d.x, 2), round(d.y, 2)) not in high_keys
    ]
    return high_candidates, recovery_candidates, high_candidates + recovery_candidates


def clamp_box(box: InitBox, width: int, height: int) -> InitBox | None:
    x1 = max(0, min(width - 1, box.x))
    y1 = max(0, min(height - 1, box.y))
    x2 = max(0, min(width, box.x + box.width))
    y2 = max(0, min(height, box.y + box.height))
    if x2 <= x1 or y2 <= y1:
        return None
    return InitBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def read_frame(video_path: Path, frame_idx: int) -> tuple[np.ndarray, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if frame_idx < 0:
        cap.release()
        raise ValueError("--init-frame must be >= 0")
    if frame_count > 0 and frame_idx >= frame_count:
        cap.release()
        raise ValueError(f"--init-frame {frame_idx} is outside the video frame range 0..{frame_count - 1}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame, fps, frame_count


def save_init_boxes(path: Path, video_path: Path, frame_idx: int, boxes: list[InitBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": str(video_path),
        "frame_idx": frame_idx,
        "boxes": [
            {"x": box.x, "y": box.y, "width": box.width, "height": box.height}
            for box in boxes
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_init_boxes(path: Path) -> tuple[int, list[InitBox]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    boxes = [
        InitBox(
            x=int(item["x"]),
            y=int(item["y"]),
            width=int(item["width"]),
            height=int(item["height"]),
        )
        for item in data.get("boxes", [])
    ]
    frame_idx = int(data.get("frame_idx", 0))
    if not boxes:
        raise ValueError(f"No boxes found in {path}")
    return frame_idx, boxes


def select_manual_boxes(video_path: Path, frame_idx: int) -> list[InitBox]:
    frame, _, _ = read_frame(video_path, frame_idx)
    display = frame.copy()
    cv2.putText(
        display,
        "Draw a box, ENTER/SPACE confirms it, ESC finishes all boxes",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    try:
        selections = cv2.selectROIs("manual target initialization", display, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("manual target initialization")
    except cv2.error as exc:
        raise RuntimeError(
            "Manual initialization requires a working OpenCV GUI display. "
            "If this machine is headless, create an init-boxes JSON file and pass --init-boxes."
        ) from exc

    boxes = []
    height, width = frame.shape[:2]
    for x, y, w, h in selections:
        if int(w) <= 0 or int(h) <= 0:
            continue
        box = clamp_box(InitBox(int(x), int(y), int(w), int(h)), width, height)
        if box is not None:
            boxes.append(box)
    if not boxes:
        raise RuntimeError("No initialization boxes were selected.")
    return boxes


def detection_inside_box(detection: Detection, box: InitBox) -> bool:
    return box.x <= detection.x <= box.x + box.width and box.y <= detection.y <= box.y + box.height


def nearest_detection_to_box(
    detections: list[Detection],
    box: InitBox,
    max_distance: float,
) -> Detection | None:
    cx = box.x + box.width / 2.0
    cy = box.y + box.height / 2.0
    inside = [detection for detection in detections if detection_inside_box(detection, box)]
    candidates = inside if inside else detections
    best: Detection | None = None
    best_distance = float("inf")
    for detection in candidates:
        distance = math.hypot(detection.x - cx, detection.y - cy)
        if distance < best_distance:
            best = detection
            best_distance = distance
    if best is not None and (inside or best_distance <= max_distance):
        return best
    return None


def brightest_detection_in_box(frame: np.ndarray, box: InitBox) -> Detection:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[box.y : box.y + box.height, box.x : box.x + box.width]
    _, value, _, location = cv2.minMaxLoc(roi)
    x = box.x + location[0]
    y = box.y + location[1]
    return Detection(
        x=float(x),
        y=float(y),
        area=1,
        width=1,
        height=1,
        intensity=float(value),
        score=float(value),
        contrast=float(value),
        source="init_brightest",
    )


def local_point_detection(
    frame: np.ndarray,
    x: float,
    y: float,
    radius: int,
    sigma: float,
    min_score: float,
) -> Detection | None:
    height, width = frame.shape[:2]
    cx = int(round(x))
    cy = int(round(y))
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(width, cx + radius + 1)
    y2 = min(height, cy + radius + 1)
    if x2 <= x1 or y2 <= y1:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[y1:y2, x1:x2]
    blur_sigma = max(1.0, min(float(sigma), max(1.0, radius / 2.0)))
    background = cv2.GaussianBlur(roi, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    highpass = cv2.subtract(roi, background)

    _, score, _, location = cv2.minMaxLoc(highpass)
    if score < min_score:
        return None
    px = x1 + location[0]
    py = y1 + location[1]
    return Detection(
        x=float(px),
        y=float(py),
        area=1,
        width=1,
        height=1,
        intensity=float(gray[py, px]),
        score=float(score),
        contrast=float(score),
        source="local",
    )


def smoothed_response(track: Track, response: float, window: int) -> float:
    if track.response_history.maxlen != window:
        previous = list(track.response_history)[-window:]
        track.response_history = deque(previous, maxlen=window)
    track.response_history.append(response)
    return float(sum(track.response_history) / len(track.response_history))


def manual_track_detections(
    frame: np.ndarray,
    tracks: list[Track],
    global_detections: list[Detection],
    local_radius: int,
    correction_radius: float,
    sigma: float,
    min_score: float,
    tbd_window: int,
    tbd_min_response: float,
) -> list[Detection]:
    detections: list[Detection] = []
    used_global: set[int] = set()
    used_points: set[tuple[int, int]] = set()

    for track in tracks:
        tx, ty = track.position
        best_idx: int | None = None
        best_distance = float("inf")
        for det_idx, detection in enumerate(global_detections):
            if det_idx in used_global:
                continue
            distance = math.hypot(tx - detection.x, ty - detection.y)
            if distance <= correction_radius and distance < best_distance:
                best_idx = det_idx
                best_distance = distance

        if best_idx is not None:
            detection = global_detections[best_idx]
            used_global.add(best_idx)
        else:
            detection = local_point_detection(frame, tx, ty, local_radius, sigma, min_score)

        if detection is None:
            smoothed_response(track, 0.0, tbd_window)
            continue
        smooth = smoothed_response(track, detection.score, tbd_window)
        if smooth < tbd_min_response:
            continue
        source = "tbd" if best_idx is None else detection.source
        detection = Detection(
            x=detection.x,
            y=detection.y,
            area=detection.area,
            width=detection.width,
            height=detection.height,
            intensity=detection.intensity,
            score=detection.score,
            contrast=smooth,
            source=source,
        )
        key = (int(round(detection.x)), int(round(detection.y)))
        if key in used_points:
            continue
        used_points.add(key)
        detections.append(detection)
    return detections


def initial_detections_from_boxes(
    video_path: Path,
    frame_idx: int,
    boxes: list[InitBox],
    ignore_mask: np.ndarray,
    args: argparse.Namespace,
) -> list[Detection]:
    frame, _, _ = read_frame(video_path, frame_idx)
    height, width = frame.shape[:2]
    detections = detect_points(
        frame=frame,
        ignore_mask=ignore_mask,
        sigma=args.sigma,
        threshold_k=args.threshold_k,
        min_threshold=args.min_threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        max_width=args.max_width,
        max_height=args.max_height,
    )

    selected: list[Detection] = []
    used: set[tuple[int, int]] = set()
    for raw_box in boxes:
        box = clamp_box(raw_box, width, height)
        if box is None:
            continue
        detection = nearest_detection_to_box(detections, box, args.init_search_radius)
        if detection is None:
            detection = brightest_detection_in_box(frame, box)
        key = (int(round(detection.x)), int(round(detection.y)))
        if key in used:
            continue
        used.add(key)
        selected.append(detection)

    if not selected:
        raise RuntimeError("Initialization boxes did not produce any target centers.")
    return selected


class MultiObjectTracker:
    def __init__(
        self,
        gate_distance: float,
        min_hits: int,
        max_misses: int,
        min_track_length: int,
        trail_length: int,
        process_noise: float,
        measurement_noise: float,
    ) -> None:
        self.gate_distance = gate_distance
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.min_track_length = min_track_length
        self.trail_length = trail_length
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.next_track_id = 1
        self.active: list[Track] = []
        self.finished: list[Track] = []

    def _new_track(self, detection: Detection) -> Track:
        state = np.array([detection.x, detection.y, 0.0, 0.0], dtype=np.float64)
        cov = np.diag([16.0, 16.0, 100.0, 100.0]).astype(np.float64)
        track = Track(
            track_id=self.next_track_id,
            state=state,
            cov=cov,
            confirmed=self.min_hits <= 1,
            last_detection=detection,
            confidence=1.0,
            response_ema=detection.score,
            last_association_stage="init",
            response_history=deque([detection.score], maxlen=5),
            trail=deque(maxlen=self.trail_length),
        )
        track.trail.append(track.position)
        self.next_track_id += 1
        return track

    def initialize(self, detections: list[Detection]) -> None:
        if self.active or self.finished:
            raise RuntimeError("Tracker can only be initialized before tracking starts.")
        for detection in detections:
            track = self._new_track(detection)
            track.hits = self.min_hits
            track.confirmed = True
            self.active.append(track)

    def _transition(self, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        transition = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        q = self.process_noise
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        process = q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )
        return transition, process

    def _predict_track(self, track: Track) -> None:
        transition, process = self._transition()
        track.state = transition @ track.state
        track.cov = transition @ track.cov @ transition.T + process
        track.age += 1

    def _update_track(self, track: Track, detection: Detection, stage: str) -> None:
        prev_x, prev_y = track.position
        measurement = np.array([detection.x, detection.y], dtype=np.float64)
        observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        measurement_cov = (self.measurement_noise**2) * np.eye(2, dtype=np.float64)

        innovation = measurement - observation @ track.state
        innovation_cov = observation @ track.cov @ observation.T + measurement_cov
        kalman_gain = track.cov @ observation.T @ np.linalg.inv(innovation_cov)

        track.state = track.state + kalman_gain @ innovation
        observed_vx = detection.x - prev_x
        observed_vy = detection.y - prev_y
        track.state[2] = 0.65 * track.state[2] + 0.35 * observed_vx
        track.state[3] = 0.65 * track.state[3] + 0.35 * observed_vy
        identity = np.eye(4, dtype=np.float64)
        track.cov = (identity - kalman_gain @ observation) @ track.cov
        track.hits += 1
        track.misses = 0
        track.last_detection = detection
        track.last_association_stage = stage
        track.response_ema = 0.8 * track.response_ema + 0.2 * detection.score
        track.confidence = min(1.0, 0.85 * track.confidence + 0.15 + min(detection.score, 50.0) / 500.0)
        if detection.score < max(1.0, self.measurement_noise):
            track.low_response_streak += 1
        else:
            track.low_response_streak = 0
        if track.hits >= self.min_hits:
            track.confirmed = True
        track.trail.append(track.position)

    def _mark_missed(self, track: Track) -> None:
        track.misses += 1
        track.last_detection = None
        track.last_association_stage = "miss"
        track.confidence = max(0.0, track.confidence * 0.85)
        track.low_response_streak += 1
        if track.confirmed:
            track.trail.append(track.position)

    def predict_all(self) -> None:
        for track in self.active:
            self._predict_track(track)

    def update(
        self,
        detections: list[Detection],
        frame_idx: int,
        time_s: float,
        allow_new_tracks: bool = True,
        already_predicted: bool = False,
        low_detections: list[Detection] | None = None,
        association: str = "standard",
        drop_lost_tracks: bool = True,
    ) -> list[Track]:
        if not already_predicted:
            self.predict_all()

        matched_tracks: set[int] = set()
        matched_high: set[int] = set()

        def match_stage(
            candidate_detections: list[Detection],
            track_indices: list[int],
            stage: str,
        ) -> set[int]:
            matched_detection_indices: set[int] = set()
            if not track_indices or not candidate_detections:
                return matched_detection_indices
            cost = np.full((len(track_indices), len(candidate_detections)), fill_value=1e6, dtype=np.float64)
            for row_idx, track_idx in enumerate(track_indices):
                track = self.active[track_idx]
                tx, ty = track.position
                for det_idx, detection in enumerate(candidate_detections):
                    distance = math.hypot(tx - detection.x, ty - detection.y)
                    if distance <= self.gate_distance:
                        response_bonus = min(detection.score, 50.0) / 100.0
                        cost[row_idx, det_idx] = distance - response_bonus

            row_indices, col_indices = linear_sum_assignment(cost)
            for row_idx, det_idx in zip(row_indices, col_indices):
                if cost[row_idx, det_idx] > self.gate_distance:
                    continue
                track_idx = track_indices[row_idx]
                if track_idx in matched_tracks:
                    continue
                self._update_track(self.active[track_idx], candidate_detections[det_idx], stage)
                matched_tracks.add(track_idx)
                matched_detection_indices.add(det_idx)
            return matched_detection_indices

        all_track_indices = list(range(len(self.active)))
        if association == "bytetrack":
            primary_stage = "high"
        elif association == "manual":
            primary_stage = "manual"
        else:
            primary_stage = "standard"
        matched_high = match_stage(detections, all_track_indices, primary_stage)
        if association == "bytetrack" and low_detections:
            remaining = [idx for idx in all_track_indices if idx not in matched_tracks]
            match_stage(low_detections, remaining, "low")

        for track_idx, track in enumerate(self.active):
            if track_idx not in matched_tracks:
                self._mark_missed(track)

        for det_idx, detection in enumerate(detections):
            if allow_new_tracks and det_idx not in matched_high:
                self.active.append(self._new_track(detection))

        kept: list[Track] = []
        for track in self.active:
            if drop_lost_tracks and track.misses > self.max_misses:
                self.finished.append(track)
                continue
            kept.append(track)
        self.active = kept

        visible_tracks = [track for track in self.active if track.confirmed]
        for track in visible_tracks:
            detection = track.last_detection
            x, y = track.position
            vx, vy = track.velocity
            if detection is None:
                state = "lost" if track.low_response_streak > self.max_misses else "predicted"
                source = ""
                stage = "miss"
                response = math.nan
            else:
                stage = track.last_association_stage
                source = detection.source
                response = detection.contrast
                state = "tbd" if detection.source == "tbd" else f"tracked_{stage}"
            track.records.append(
                TrackRecord(
                    frame_idx=frame_idx,
                    time_s=time_s,
                    track_id=track.track_id,
                    x=x,
                    y=y,
                    vx=vx,
                    vy=vy,
                    intensity=detection.intensity if detection is not None else math.nan,
                    score=detection.score if detection is not None else math.nan,
                    confidence=track.confidence,
                    response=response,
                    source=source,
                    association_stage=stage,
                    state=state,
                )
            )
        return visible_tracks

    def finish(self) -> list[Track]:
        self.finished.extend(self.active)
        self.active = []
        return [track for track in self.finished if track.confirmed and track.hits >= self.min_track_length]

    def apply_translation(self, dx: float, dy: float) -> None:
        for track in self.active:
            track.state[0] += dx
            track.state[1] += dy


def color_for_track(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 37) % 180
    hsv = np.array([[[hue, 210, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    detections: list[Detection],
    frame_idx: int,
    draw_detections: bool,
) -> np.ndarray:
    canvas = frame.copy()
    if draw_detections:
        for detection in detections:
            center = (int(round(detection.x)), int(round(detection.y)))
            cv2.drawMarker(canvas, center, (180, 180, 180), cv2.MARKER_CROSS, 7, 1)

    for track in tracks:
        color = color_for_track(track.track_id)
        points = [(int(round(x)), int(round(y))) for x, y in track.trail]
        if len(points) >= 2:
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
        x, y = track.position
        center = (int(round(x)), int(round(y)))
        cv2.circle(canvas, center, 4, color, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(track.track_id),
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"frame {frame_idx} tracks {len(tracks)}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return canvas


def write_tracks_csv(csv_path: Path, tracks: list[Track]) -> int:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_idx",
        "time_s",
        "track_id",
        "x",
        "y",
        "vx",
        "vy",
        "intensity",
        "score",
        "confidence",
        "response",
        "source",
        "association_stage",
        "state",
    ]
    row_count = 0
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for track in sorted(tracks, key=lambda item: item.track_id):
            for record in track.records:
                writer.writerow(
                    {
                        "frame_idx": record.frame_idx,
                        "time_s": f"{record.time_s:.6f}",
                        "track_id": record.track_id,
                        "x": f"{record.x:.3f}",
                        "y": f"{record.y:.3f}",
                        "vx": f"{record.vx:.3f}",
                        "vy": f"{record.vy:.3f}",
                        "intensity": "" if math.isnan(record.intensity) else f"{record.intensity:.1f}",
                        "score": "" if math.isnan(record.score) else f"{record.score:.1f}",
                        "confidence": f"{record.confidence:.3f}",
                        "response": "" if math.isnan(record.response) else f"{record.response:.3f}",
                        "source": record.source,
                        "association_stage": record.association_stage,
                        "state": record.state,
                    }
                )
                row_count += 1
    return row_count


class DebugDetectionWriter:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.handle = None
        self.writer: csv.DictWriter | None = None

    def __enter__(self) -> "DebugDetectionWriter":
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("w", newline="")
            self.writer = csv.DictWriter(
                self.handle,
                fieldnames=["frame_idx", "time_s", "x", "y", "score", "contrast", "intensity", "area", "source"],
            )
            self.writer.writeheader()
        return self

    def write(self, frame_idx: int, time_s: float, detections: list[Detection]) -> None:
        if self.writer is None:
            return
        for detection in detections:
            self.writer.writerow(
                {
                    "frame_idx": frame_idx,
                    "time_s": f"{time_s:.6f}",
                    "x": f"{detection.x:.3f}",
                    "y": f"{detection.y:.3f}",
                    "score": f"{detection.score:.3f}",
                    "contrast": f"{detection.contrast:.3f}",
                    "intensity": f"{detection.intensity:.1f}",
                    "area": detection.area,
                    "source": detection.source,
                }
            )

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.handle is not None:
            self.handle.close()


def estimate_phase_translation(prev_gray: np.ndarray | None, frame: np.ndarray) -> tuple[float, float, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        return 0.0, 0.0, gray
    shift, response = cv2.phaseCorrelate(np.float32(prev_gray), np.float32(gray))
    if not np.isfinite(response) or response < 0.05:
        return 0.0, 0.0, gray
    dx, dy = shift
    if abs(dx) > 20 or abs(dy) > 20:
        return 0.0, 0.0, gray
    return float(dx), float(dy), gray


def track_video(video_path: Path, vis_path: Path, csv_path: Path, args: argparse.Namespace) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for tracking: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or math.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read video dimensions from {video_path}")

    vis_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(vis_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open visualization writer: {vis_path}")

    ignore_mask = build_ignore_mask(height, width, args.ignore_corner_size)
    tracker = MultiObjectTracker(
        gate_distance=args.gate_distance,
        min_hits=args.min_hits,
        max_misses=args.max_misses,
        min_track_length=args.min_track_length,
        trail_length=args.trail_length,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
    )

    init_frame_idx: int | None = None
    init_boxes: list[InitBox] = []
    if args.init_boxes is not None:
        init_frame_idx, init_boxes = load_init_boxes(args.init_boxes)
    elif args.manual_init:
        init_frame_idx = args.init_frame
        init_boxes = select_manual_boxes(video_path, init_frame_idx)
        if args.save_init_boxes is not None:
            save_init_boxes(args.save_init_boxes, video_path, init_frame_idx, init_boxes)
            print(f"Saved {len(init_boxes)} initialization boxes -> {args.save_init_boxes}")
    elif args.save_init_boxes is not None:
        raise ValueError("--save-init-boxes requires --manual-init")

    manual_tracking = bool(init_boxes)
    if manual_tracking:
        if init_frame_idx is None:
            init_frame_idx = args.init_frame
        if frame_count > 0 and init_frame_idx >= frame_count:
            raise ValueError(f"Initialization frame {init_frame_idx} is outside the video frame range 0..{frame_count - 1}")
        manual_ignore_mask = np.zeros_like(ignore_mask)
        init_detections = initial_detections_from_boxes(video_path, init_frame_idx, init_boxes, manual_ignore_mask, args)
        tracker.min_track_length = min(tracker.min_track_length, tracker.min_hits)
        tracker.initialize(init_detections)
        cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame_idx)
        print(f"Initialized {len(init_detections)} manually selected target(s) at frame {init_frame_idx}")
        frame_idx = init_frame_idx
    else:
        manual_ignore_mask = ignore_mask
        frame_idx = 0

    prev_gray: np.ndarray | None = None
    with DebugDetectionWriter(args.debug_detections) as debug_writer:
        with tqdm(total=frame_count if frame_count > 0 else None, desc="tracking", unit="frame") as progress:
            if manual_tracking and init_frame_idx:
                progress.update(init_frame_idx)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                time_s = frame_idx / fps
                if args.motion_comp == "phase":
                    dx, dy, prev_gray = estimate_phase_translation(prev_gray, frame)
                    if dx or dy:
                        tracker.apply_translation(dx, dy)

                high_detections, low_detections, all_candidates = detect_tiered_points(
                    frame=frame,
                    ignore_mask=manual_ignore_mask,
                    sigma=args.sigma,
                    high_threshold=args.high_threshold,
                    low_threshold=args.low_threshold,
                    high_percentile=args.high_percentile,
                    min_area=args.min_area,
                    max_area=args.max_area,
                    max_width=args.max_width,
                    max_height=args.max_height,
                )
                detections = high_detections
                debug_candidates = all_candidates
                already_predicted = False
                if manual_tracking:
                    tracker.predict_all()
                    already_predicted = True
                    manual_global_detections = high_detections if args.manual_global_correction else []
                    detections = manual_track_detections(
                        frame=frame,
                        tracks=tracker.active,
                        global_detections=manual_global_detections,
                        local_radius=args.manual_search_radius,
                        correction_radius=args.manual_correction_radius,
                        sigma=args.sigma,
                        min_score=args.manual_min_score,
                        tbd_window=args.tbd_window,
                        tbd_min_response=args.tbd_min_response,
                    )
                    low_detections = []
                    debug_candidates = all_candidates + detections
                visible_tracks = tracker.update(
                    detections,
                    frame_idx,
                    time_s,
                    allow_new_tracks=not manual_tracking,
                    already_predicted=already_predicted,
                    low_detections=low_detections,
                    association="manual" if manual_tracking else args.association,
                    drop_lost_tracks=(not manual_tracking or args.manual_drop_lost),
                )
                debug_writer.write(frame_idx, time_s, debug_candidates)
                canvas = draw_tracks(frame, visible_tracks, detections, frame_idx, args.draw_detections)
                writer.write(canvas)
                frame_idx += 1
                progress.update(1)

    cap.release()
    writer.release()

    finished_tracks = tracker.finish()
    row_count = write_tracks_csv(csv_path, finished_tracks)
    return frame_idx, row_count


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input video does not exist: {args.input}")

    clip_out = args.clip_out or default_clip_path(args.input, args.clip_seconds)
    if args.skip_clip:
        track_input = args.input
        print(f"Skipping clip step; tracking {track_input}")
    else:
        start_s, keep_s = clip_last_seconds(args.input, clip_out, args.clip_seconds)
        track_input = clip_out
        print(f"Clipped last {keep_s:.3f}s from {args.input} starting at {start_s:.3f}s -> {clip_out}")

    vis_out = args.vis_out or default_vis_path(track_input)
    frame_count, row_count = track_video(track_input, vis_out, args.csv_out, args)
    print(f"Tracked {frame_count} frames -> {vis_out}")
    print(f"Wrote {row_count} CSV rows -> {args.csv_out}")


if __name__ == "__main__":
    main()

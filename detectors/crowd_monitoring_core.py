"""Crowd panic monitoring: person detection, tracking, optical flow, panic scoring.

Shared by CrowdMonitoringDetector and standalone benchmark scripts.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PERSON_CLASS = 0

SPEED_SPIKE_THRESH = 18.0
DENSITY_CRITICAL = 35
CONVERGENCE_THRESH = 0.60
TRAIL_LEN = 24


@dataclass
class CrowdMetrics:
    spike_ratio: float = 0.0
    dir_chaos: float = 0.0
    motion_entropy: float = 0.0
    convergence: float = 0.0
    density_pressure: float = 0.0
    avg_speed: float = 0.0


@dataclass
class CrowdFrameResult:
    crowd_count: int = 0
    panic_score: float = 0.0
    panic_state: str = "NORMAL"
    gated_state: str = "NORMAL"
    metrics: CrowdMetrics = field(default_factory=CrowdMetrics)
    person_bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    latency_ms: float = 0.0


class PersonTrack:
    def __init__(self, tid, bbox):
        self.id = tid
        self.bbox = bbox
        self.hits = 1
        self.misses = 0
        self.history = collections.deque(maxlen=TRAIL_LEN)
        self.vel_history = collections.deque(maxlen=12)
        self.speed = 0.0
        cx, cy = self._center(bbox)
        self.history.append((cx, cy))

    @staticmethod
    def _center(b):
        return int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)

    def update(self, bbox):
        px, py = self.history[-1]
        cx, cy = self._center(bbox)
        dx, dy = cx - px, cy - py
        spd = math.hypot(dx, dy)
        self.vel_history.append((dx, dy, spd))
        self.speed = spd
        self.bbox = bbox
        self.history.append((cx, cy))
        self.hits += 1
        self.misses = 0

    @property
    def avg_speed(self):
        if not self.vel_history:
            return 0.0
        return float(np.mean([v[2] for v in self.vel_history]))

    @property
    def direction_variance(self):
        if len(self.vel_history) < 3:
            return 0.0
        dirs = [math.atan2(v[1], v[0]) for v in self.vel_history if v[2] > 1]
        if len(dirs) < 2:
            return 0.0
        diffs = [abs(math.sin(dirs[i] - dirs[i - 1])) for i in range(1, len(dirs))]
        return float(np.mean(diffs))


class IoUTracker:
    def __init__(self, iou_thresh=0.30, max_misses=6):
        self.tracks = {}
        self.next_id = 1
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses

    @staticmethod
    def _iou(a, b):
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0

    def update(self, bboxes):
        matched_t, matched_d = set(), set()
        bboxes = list(bboxes)
        for tid, t in self.tracks.items():
            best_iou, best_di = 0, -1
            for di, bb in enumerate(bboxes):
                if di in matched_d:
                    continue
                iou = self._iou(t.bbox, bb)
                if iou > best_iou:
                    best_iou, best_di = iou, di
            if best_iou >= self.iou_thresh:
                t.update(bboxes[best_di])
                matched_t.add(tid)
                matched_d.add(best_di)
        for tid in list(self.tracks):
            if tid not in matched_t:
                self.tracks[tid].misses += 1
        for di, bb in enumerate(bboxes):
            if di not in matched_d:
                self.tracks[self.next_id] = PersonTrack(self.next_id, bb)
                self.next_id += 1
        self.tracks = {t: v for t, v in self.tracks.items() if v.misses <= self.max_misses}
        return self.tracks


class OpticalFlowEngine:
    def __init__(self):
        self.prev_gray = None
        self.flow = None

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        self.flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=12,
            iterations=3,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )
        self.prev_gray = gray
        return self.flow

    def compute_convergence(self):
        if self.flow is None:
            return 0.0
        dx = self.flow[:, :, 0]
        dy = self.flow[:, :, 1]
        div_x = np.gradient(dx, axis=1)
        div_y = np.gradient(dy, axis=0)
        divergence = div_x + div_y
        return float(np.clip(-np.mean(divergence) * 5, 0, 1))

    def compute_motion_entropy(self):
        if self.flow is None:
            return 0.0
        dx = self.flow[:, :, 0].ravel()
        dy = self.flow[:, :, 1].ravel()
        mag = np.hypot(dx, dy)
        mask = mag > 0.5
        if mask.sum() < 10:
            return 0.0
        angles = np.arctan2(dy[mask], dx[mask])
        hist, _ = np.histogram(angles, bins=16, range=(-np.pi, np.pi), density=True)
        hist += 1e-9
        entropy = -np.sum(hist * np.log(hist + 1e-9))
        return float(np.clip(entropy / math.log(16), 0, 1))


class PanicDetector:
    def __init__(self):
        self.panic_history = collections.deque(maxlen=20)

    def analyze(self, tracks, flow_engine, crowd_count):
        speeds = [t.avg_speed for t in tracks.values()]
        if speeds:
            spike_ratio = sum(s > SPEED_SPIKE_THRESH for s in speeds) / len(speeds)
            avg_spd = float(np.mean(speeds))
        else:
            spike_ratio, avg_spd = 0.0, 0.0

        dir_vars = [t.direction_variance for t in tracks.values()]
        dir_chaos = float(np.mean(dir_vars)) if dir_vars else 0.0
        motion_entropy = flow_engine.compute_motion_entropy()
        convergence = flow_engine.compute_convergence()
        density_pressure = min(1.0, crowd_count / max(DENSITY_CRITICAL, 1))

        panic_score = (
            spike_ratio * 0.25
            + dir_chaos * 0.25
            + motion_entropy * 0.20
            + convergence * 0.15
            + density_pressure * 0.15
        )
        panic_score = float(np.clip(panic_score, 0.0, 1.0))
        self.panic_history.append(panic_score)
        smoothed = float(np.mean(self.panic_history))

        if smoothed < 0.25:
            state = "NORMAL"
        elif smoothed < 0.50:
            state = "ALERT"
        elif smoothed < 0.72:
            state = "PANIC RISK"
        else:
            state = "MASS PANIC"

        metrics = CrowdMetrics(
            spike_ratio=spike_ratio,
            dir_chaos=dir_chaos,
            motion_entropy=motion_entropy,
            convergence=convergence,
            density_pressure=density_pressure,
            avg_speed=avg_spd,
        )
        return state, smoothed, metrics


class SustainedAlertGate:
    def __init__(self, sustain_sec=3.0):
        self._sustain = sustain_sec
        self._first_seen = {}
        self._current = "NORMAL"
        self._confirmed = "NORMAL"

    def update(self, state):
        import time

        now = time.time()
        if state == "NORMAL":
            self._first_seen = {}
            self._current = "NORMAL"
            self._confirmed = "NORMAL"
            return "NORMAL"
        if state != self._current:
            self._first_seen = {state: now}
            self._current = state
            return self._confirmed
        first = self._first_seen.get(state, now)
        if (now - first) >= self._sustain:
            self._confirmed = state
        return self._confirmed


class CrowdFrameProcessor:
    """One-frame crowd panic pipeline (YOLO persons + track + flow + panic score)."""

    def __init__(
        self,
        model_path="yolov8n.pt",
        confidence=0.30,
        infer_every=2,
        infer_size=416,
        flow_every=2,
        sustain_sec=3.0,
        device=None,
    ):
        self.confidence = confidence
        self.infer_every = infer_every
        self.infer_size = infer_size
        self.flow_every = flow_every
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = self.device == "cuda"

        self.model = YOLO(model_path)
        if self.use_fp16:
            self.model.model.half()

        self.tracker = IoUTracker()
        self.flow_engine = OpticalFlowEngine()
        self.panic_det = PanicDetector()
        self.alert_gate = SustainedAlertGate(sustain_sec=sustain_sec)

        self.frame_idx = 0
        self._last_bboxes: List[Tuple[int, int, int, int]] = []
        self._last_panic = (0.0, "NORMAL", CrowdMetrics())
        self._frame_h = 0
        self._frame_w = 0

    def detect_persons(self, frame) -> List[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.infer_size, self.infer_size))
        results = self.model(
            small,
            verbose=False,
            device=self.device,
            half=self.use_fp16,
            conf=self.confidence,
            iou=0.45,
            classes=[PERSON_CLASS],
        )
        sx = w / self.infer_size
        sy = h / self.infer_size
        bboxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bboxes.append((int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)))
        return bboxes

    def process(self, frame, measure_latency=True) -> CrowdFrameResult:
        import time

        t0 = time.perf_counter() if measure_latency else None

        self.frame_idx += 1
        h, w = frame.shape[:2]
        self._frame_h, self._frame_w = h, w

        do_infer = self.frame_idx % self.infer_every == 0
        do_flow = self.frame_idx % self.flow_every == 0

        if do_flow:
            self.flow_engine.update(frame)

        if do_infer:
            self._last_bboxes = self.detect_persons(frame)

        tracks = self.tracker.update(self._last_bboxes)
        crowd_count = len(tracks)

        if do_infer:
            panic_state, panic_score, metrics = self.panic_det.analyze(
                tracks, self.flow_engine, crowd_count
            )
            self._last_panic = (panic_score, panic_state, metrics)
        else:
            panic_score, panic_state, metrics = self._last_panic

        gated_state = self.alert_gate.update(panic_state)

        latency_ms = 0.0
        if t0 is not None:
            latency_ms = (time.perf_counter() - t0) * 1000

        return CrowdFrameResult(
            crowd_count=crowd_count,
            panic_score=panic_score,
            panic_state=panic_state,
            gated_state=gated_state,
            metrics=metrics,
            person_bboxes=list(self._last_bboxes),
            latency_ms=latency_ms,
        )

    def draw_overlay(self, frame, result: CrowdFrameResult):
        out = frame.copy()
        for x1, y1, x2, y2 in result.person_bboxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), (57, 255, 20), 2)
        label = (
            f"Crowd:{result.crowd_count}  Panic:{result.panic_score:.0%}  "
            f"{result.gated_state}"
        )
        cv2.rectangle(out, (0, 0), (out.shape[1], 28), (6, 10, 18), -1)
        cv2.putText(
            out,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return out

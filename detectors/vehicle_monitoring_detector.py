"""Vehicle monitoring with YOLOv8n detection and ByteTrack multi-object tracking.

Monitors configured lane/ROI polygons for entry, exit, movement, and obstruction
(dwell-based). Events are sent as JSON through the event and alert engines.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import config
from detectors.base_detector import BaseDetector
from engines.alert_engine import AlertType

# COCO indices for vehicle classes in the default YOLOv8n weights.
VEHICLE_CLASS_IDS = (2, 3, 5, 7)


class _TrackState:
    def __init__(self, track_id, vehicle_class, confidence):
        self.track_id = track_id
        self.vehicle_class = vehicle_class
        self.confidence = confidence
        self.lanes = set()
        self.lane_entered_at = {}
        self.lane_obstruction_sent = set()
        self.last_center = None
        self.last_movement_ref = None
        self.last_movement_event_at = 0.0


class VehicleMonitoringDetector(BaseDetector):
    """Detects vehicles, tracks them with ByteTrack, and raises lane/ROI events."""

    def __init__(
        self,
        model_path=None,
        lanes_config_path=None,
        confidence=None,
        obstruction_dwell_seconds=None,
        movement_threshold_px=None,
        movement_cooldown_seconds=None,
    ):
        super().__init__()
        self.model_path = model_path or config.VEHICLE_MODEL_PATH
        self.lanes_config_path = lanes_config_path or config.VEHICLE_LANES_CONFIG_PATH
        self.confidence = confidence if confidence is not None else config.VEHICLE_CONFIDENCE
        self.obstruction_dwell_seconds = (
            obstruction_dwell_seconds
            if obstruction_dwell_seconds is not None
            else config.VEHICLE_OBSTRUCTION_DWELL_SECONDS
        )
        self.movement_threshold_px = (
            movement_threshold_px
            if movement_threshold_px is not None
            else config.VEHICLE_MOVEMENT_THRESHOLD_PX
        )
        self.movement_cooldown_seconds = (
            movement_cooldown_seconds
            if movement_cooldown_seconds is not None
            else config.VEHICLE_MOVEMENT_COOLDOWN_SECONDS
        )

        self._model = None
        self._lanes = []
        self._tracks = {}

    def on_start(self):
        self._model = YOLO(self.model_path)
        self._lanes = self._load_lanes(self.lanes_config_path)
        self.log(f"model loaded from {self.model_path}")
        if self._lanes:
            lane_ids = ", ".join(lane["id"] for lane in self._lanes)
            self.log(f"monitoring {len(self._lanes)} lane ROI(s): {lane_ids}")
        else:
            self.log(
                "no lane ROIs configured; vehicle tracks are updated but lane events are disabled",
                level="WARNING",
            )

    def process(self, frame):
        result = self._model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            classes=list(VEHICLE_CLASS_IDS),
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            self._finalize_missing_tracks(set(), frame, result)
            return

        seen_track_ids = set()
        pending_events = []

        for box in result.boxes:
            if box.id is None:
                continue

            track_id = int(box.id.item())
            class_id = int(box.cls.item())
            vehicle_class = result.names[class_id]
            confidence = float(box.conf.item())
            center = self._bottom_center(box)

            seen_track_ids.add(track_id)
            state = self._tracks.get(track_id)
            if state is None:
                state = _TrackState(track_id, vehicle_class, confidence)
                self._tracks[track_id] = state

            state.vehicle_class = vehicle_class
            state.confidence = confidence
            lanes_now = self._lanes_for_point(center)
            pending_events.extend(self._lane_transition_events(state, lanes_now, center))
            pending_events.extend(self._movement_events(state, center, lanes_now))
            pending_events.extend(self._obstruction_events(state, center, lanes_now))
            state.last_center = center

        exit_events = self._finalize_missing_tracks(seen_track_ids, frame, result)
        pending_events.extend(exit_events)

        if pending_events:
            snapshot = self._annotate_frame(frame, result)
            for event in pending_events:
                self._emit_event(snapshot, **event)

    def _finalize_missing_tracks(self, seen_track_ids, frame, result):
        stale_ids = [track_id for track_id in self._tracks if track_id not in seen_track_ids]
        events = []
        for track_id in stale_ids:
            state = self._tracks.pop(track_id)
            for lane_id in sorted(state.lanes):
                events.append(self._event_payload("exit", state, lane_id))
        return events

    def _lane_transition_events(self, state, lanes_now, center):
        events = []
        entered = lanes_now - state.lanes
        exited = state.lanes - lanes_now

        for lane_id in sorted(entered):
            state.lane_entered_at[lane_id] = time.time()
            state.lane_obstruction_sent.discard(lane_id)
            state.last_movement_ref = center
            events.append(
                self._event_payload("entry", state, lane_id),
            )

        for lane_id in sorted(exited):
            state.lane_entered_at.pop(lane_id, None)
            state.lane_obstruction_sent.discard(lane_id)
            events.append(
                self._event_payload("exit", state, lane_id),
            )

        state.lanes = set(lanes_now)
        return events

    def _movement_events(self, state, center, lanes_now):
        if not lanes_now or state.last_movement_ref is None:
            return []

        distance = self._distance(center, state.last_movement_ref)
        if distance < self.movement_threshold_px:
            return []

        now = time.time()
        if now - state.last_movement_event_at < self.movement_cooldown_seconds:
            state.last_movement_ref = center
            return []

        state.last_movement_event_at = now
        state.last_movement_ref = center
        lane_id = self._primary_lane(lanes_now)
        return [self._event_payload("movement", state, lane_id)]

    def _obstruction_events(self, state, center, lanes_now):
        events = []
        now = time.time()

        for lane_id in sorted(lanes_now):
            if lane_id in state.lane_obstruction_sent:
                continue

            entered_at = state.lane_entered_at.get(lane_id)
            if entered_at is None:
                continue

            dwell = now - entered_at
            if dwell < self.obstruction_dwell_seconds:
                continue

            if state.last_movement_ref is not None:
                drift = self._distance(center, state.last_movement_ref)
                if drift > self.movement_threshold_px / 2:
                    state.last_movement_ref = center
                    continue

            state.lane_obstruction_sent.add(lane_id)
            events.append(self._event_payload("obstruction", state, lane_id))

        return events

    def _event_payload(self, event_type, state, lane_id):
        return {
            "event_type": event_type,
            "track_id": state.track_id,
            "vehicle_class": state.vehicle_class,
            "lane": lane_id,
            "confidence": state.confidence,
        }

    def _emit_event(self, frame, event_type, track_id, vehicle_class, lane, confidence):
        payload = {
            "event": event_type,
            "vehicle_id": track_id,
            "class": vehicle_class,
            "lane": lane,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "confidence": round(confidence, 4),
        }
        message = json.dumps(payload, separators=(",", ":"))
        alert_type = AlertType.WARNING if event_type == "obstruction" else AlertType.NORMAL
        image_base64 = self.to_base64(frame) if frame is not None else None
        self.alert(image_base64, alert_type, message)
        self.log(f"{event_type}: vehicle {track_id} ({vehicle_class}) lane {lane}")

    def _load_lanes(self, config_path):
        path = Path(config_path)
        if not path.is_file():
            self.log(f"lane config not found at {path}", level="WARNING")
            return []

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            self.log(f"could not read lane config: {error}", level="ERROR")
            return []

        lanes = []
        for index, lane in enumerate(data.get("lanes", [])):
            polygon = lane.get("polygon") or []
            if len(polygon) < 3:
                self.log(f"skipping lane index {index}: polygon needs at least 3 points", level="WARNING")
                continue

            lane_id = lane.get("id") or lane.get("label") or f"lane_{index + 1}"
            lanes.append(
                {
                    "id": str(lane_id),
                    "label": str(lane.get("label") or lane_id),
                    "polygon": polygon,
                }
            )
        return lanes

    def _lanes_for_point(self, point):
        if not self._lanes:
            return set()

        hits = set()
        for lane in self._lanes:
            polygon = np.array(lane["polygon"], dtype=np.int32)
            # Positive distance means inside the polygon.
            if cv2.pointPolygonTest(polygon, point, False) >= 0:
                hits.add(lane["id"])
        return hits

    @staticmethod
    def _primary_lane(lanes_now):
        return sorted(lanes_now)[0]

    @staticmethod
    def _bottom_center(box):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        return ((x1 + x2) / 2, y2)

    @staticmethod
    def _distance(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _annotate_frame(self, frame, result):
        annotated = result.plot() if result is not None else frame.copy()
        for lane in self._lanes:
            polygon = np.array(lane["polygon"], dtype=np.int32)
            cv2.polylines(annotated, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)
            anchor = polygon[0]
            cv2.putText(
                annotated,
                lane["label"],
                (int(anchor[0]), int(anchor[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return annotated

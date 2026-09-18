"""Crowd density and panic-risk monitoring (person YOLO + motion / panic scoring)."""

import json
import time

import config
from detectors.base_detector import BaseDetector
from detectors.crowd_monitoring_core import CrowdFrameProcessor
from engines.alert_engine import AlertType


class CrowdMonitoringDetector(BaseDetector):
    """Logs crowd metrics and raises alerts when gated panic risk is sustained."""

    ALERT_STATES = frozenset({"PANIC RISK", "MASS PANIC"})

    def __init__(
        self,
        model_path=None,
        confidence=None,
        infer_every=None,
        infer_size=None,
        flow_every=None,
        sustain_sec=None,
        alert_cooldown=None,
        log_every_n_frames=None,
    ):
        super().__init__()
        self.model_path = model_path or config.CROWD_MODEL_PATH
        self.confidence = confidence if confidence is not None else config.CROWD_CONFIDENCE
        self.infer_every = infer_every if infer_every is not None else config.CROWD_INFER_EVERY
        self.infer_size = infer_size if infer_size is not None else config.CROWD_INFER_SIZE
        self.flow_every = flow_every if flow_every is not None else config.CROWD_FLOW_EVERY
        self.sustain_sec = sustain_sec if sustain_sec is not None else config.CROWD_SUSTAIN_SEC
        self.alert_cooldown = (
            alert_cooldown if alert_cooldown is not None else config.CROWD_ALERT_COOLDOWN
        )
        self.log_every_n_frames = (
            log_every_n_frames
            if log_every_n_frames is not None
            else config.CROWD_LOG_EVERY_N_FRAMES
        )

        self._processor = None
        self._last_alert_at = {}
        self._last_gated = "NORMAL"

    def on_start(self):
        self._processor = CrowdFrameProcessor(
            model_path=self.model_path,
            confidence=self.confidence,
            infer_every=self.infer_every,
            infer_size=self.infer_size,
            flow_every=self.flow_every,
            sustain_sec=self.sustain_sec,
        )
        self.log(
            f"crowd model loaded from {self.model_path} "
            f"(infer every {self.infer_every} frame(s), size {self.infer_size})"
        )

    def process(self, frame):
        result = self._processor.process(frame)

        if self._processor.frame_idx % self.log_every_n_frames == 0:
            self.log(
                json.dumps(
                    {
                        "crowd_count": result.crowd_count,
                        "panic_score": round(result.panic_score, 4),
                        "panic_state": result.panic_state,
                        "gated_state": result.gated_state,
                        "avg_speed": round(result.metrics.avg_speed, 2),
                        "entropy": round(result.metrics.motion_entropy, 3),
                        "convergence": round(result.metrics.convergence, 3),
                    }
                ),
                level="DEBUG",
            )

        if result.gated_state not in self.ALERT_STATES:
            self._last_gated = result.gated_state
            return

        state_changed = result.gated_state != self._last_gated
        now = time.time()
        if not state_changed:
            if now - self._last_alert_at.get(result.gated_state, 0) < self.alert_cooldown:
                return

        self._last_gated = result.gated_state
        self._last_alert_at[result.gated_state] = now
        alert_type = (
            AlertType.CRITICAL if result.gated_state == "MASS PANIC" else AlertType.WARNING
        )
        overlay = self._processor.draw_overlay(frame, result)
        message = (
            f"{result.gated_state}: count={result.crowd_count}, "
            f"panic={result.panic_score:.0%}, "
            f"entropy={result.metrics.motion_entropy:.2f}, "
            f"convergence={result.metrics.convergence:.2f}"
        )
        self.alert(self.to_base64(overlay), alert_type, message)

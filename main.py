"""Entry point: reads the camera once and feeds every attached detection in parallel.

To add a detection:
  1. create a class in detectors/ that extends BaseDetector
  2. add one DetectionWrapper line to `build_detections()` below
"""

import time

import config
from console import say
from detectors.smoke_and_fire_detector import SmokeAndFireDetector
from engines.alert_engine import AlertEngine
from engines.event_engine import EventEngine
from services.DetectionWrapper import DetectionWrapper
from services.RTSPService import RTSPService


def build_detections():
    """Return the detections to run. DetectionWrapper(name, fps, detector)."""
    return [
        DetectionWrapper("smoke_and_fire_detection", 2, SmokeAndFireDetector()),
    ]


def main():
    if not config.RTSP_URL:
        say("RTSP_URL is missing, set it in the .env file")
        return

    alert_engine = AlertEngine(
        smtp_config={
            "host": config.SMTP_HOST,
            "port": config.SMTP_PORT,
            "username": config.SMTP_USERNAME,
            "password": config.SMTP_PASSWORD,
            "from_email": config.ALERT_FROM_EMAIL,
            "to_emails": config.ALERT_TO_EMAILS,
        }
    )
    event_engine = EventEngine(config.LOG_FILE_PATH, alert_engine=alert_engine)
    reader = RTSPService(config.RTSP_URL, reconnect_delay=config.RECONNECT_DELAY)

    detections = build_detections()
    if not detections:
        say("No detection attached yet, add one in build_detections() in main.py")
        return

    alert_engine.start()
    event_engine.start()
    reader.start()

    # Each wrapper runs in its own thread, so all detections work at the same time.
    for detection in detections:
        detection.start(reader=reader, event_engine=event_engine)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        say("\nshutting down...")
    finally:
        for detection in detections:
            detection.stop()
        reader.stop()
        event_engine.stop()
        alert_engine.stop()


if __name__ == "__main__":
    main()

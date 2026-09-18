"""All settings are read from the .env file so nothing is hardcoded in the code."""

import os

from dotenv import load_dotenv

load_dotenv()

RTSP_URL = os.getenv("RTSP_URL", "")
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/events.log")
RECONNECT_DELAY = int(os.getenv("RECONNECT_DELAY", "5"))

SMOKE_FIRE_MODEL_PATH = os.getenv("SMOKE_FIRE_MODEL_PATH", "models/fire_smoke_yolov8n.pt")

VEHICLE_MODEL_PATH = os.getenv("VEHICLE_MODEL_PATH", "yolov8n.pt")
VEHICLE_LANES_CONFIG_PATH = os.getenv("VEHICLE_LANES_CONFIG_PATH", "config/vehicle_lanes.json")
VEHICLE_MONITORING_FPS = int(os.getenv("VEHICLE_MONITORING_FPS", "5"))
VEHICLE_CONFIDENCE = float(os.getenv("VEHICLE_CONFIDENCE", "0.35"))
VEHICLE_OBSTRUCTION_DWELL_SECONDS = float(os.getenv("VEHICLE_OBSTRUCTION_DWELL_SECONDS", "45"))
VEHICLE_MOVEMENT_THRESHOLD_PX = float(os.getenv("VEHICLE_MOVEMENT_THRESHOLD_PX", "40"))
VEHICLE_MOVEMENT_COOLDOWN_SECONDS = float(os.getenv("VEHICLE_MOVEMENT_COOLDOWN_SECONDS", "3"))

CROWD_MODEL_PATH = os.getenv("CROWD_MODEL_PATH", "yolov8n.pt")
CROWD_MONITORING_FPS = int(os.getenv("CROWD_MONITORING_FPS", "5"))
CROWD_CONFIDENCE = float(os.getenv("CROWD_CONFIDENCE", "0.30"))
CROWD_INFER_EVERY = int(os.getenv("CROWD_INFER_EVERY", "2"))
CROWD_INFER_SIZE = int(os.getenv("CROWD_INFER_SIZE", "416"))
CROWD_FLOW_EVERY = int(os.getenv("CROWD_FLOW_EVERY", "2"))
CROWD_SUSTAIN_SEC = float(os.getenv("CROWD_SUSTAIN_SEC", "3"))
CROWD_ALERT_COOLDOWN = float(os.getenv("CROWD_ALERT_COOLDOWN", "60"))
CROWD_LOG_EVERY_N_FRAMES = int(os.getenv("CROWD_LOG_EVERY_N_FRAMES", "30"))

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_FROM_EMAIL = os.getenv("ALERT_FROM_EMAIL", "")

# Comma separated list in .env -> list of addresses here
ALERT_TO_EMAILS = [
    email.strip() for email in os.getenv("ALERT_TO_EMAILS", "").split(",") if email.strip()
]

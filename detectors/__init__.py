from detectors.base_detector import BaseDetector
# from detectors.smoke_and_fire_detector import SmokeAndFireDetector
from detectors.crowd_monitoring_detector import CrowdMonitoringDetector
from detectors.vehicle_monitoring_detector import VehicleMonitoringDetector

__all__ = [
    "BaseDetector",
    "VehicleMonitoringDetector",
    "CrowdMonitoringDetector",
]  # , "SmokeAndFireDetector"

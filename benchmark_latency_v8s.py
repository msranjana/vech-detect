import time
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "yolov8s.pt"
VIDEO_PATH = "test_videos/Traffic Control CCTV.mp4"

NUM_FRAMES = 100
CONFIDENCE = 0.35
IMAGE_SIZE = 640

print("=" * 60)
print("YOLOv8n LATENCY BENCHMARK")
print("=" * 60)

print("\nLoading model...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

print("Video opened successfully")

ret, frame = cap.read()

if not ret:
    raise RuntimeError("Could not read video")

print(f"Frame resolution: {frame.shape[1]}x{frame.shape[0]}")

# Warmup
print("\nWarming up model...")

for _ in range(5):
    model.predict(
        frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )

print("Warmup complete")

# Benchmark
print(f"\nTesting {NUM_FRAMES} frames...\n")

latencies = []
detection_counts = []

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for frame_number in range(1, NUM_FRAMES + 1):

    ret, frame = cap.read()

    if not ret:
        print("Video ended early")
        break

    start = time.perf_counter()

    results = model.predict(
        frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )

    end = time.perf_counter()

    latency_ms = (end - start) * 1000

    latencies.append(latency_ms)

    result = results[0]

    if result.boxes is not None:
        count = len(result.boxes)
    else:
        count = 0

    detection_counts.append(count)

    if frame_number % 10 == 0:
        print(
            f"Frame {frame_number:3d} | "
            f"Latency: {latency_ms:8.2f} ms | "
            f"Detections: {count}"
        )

cap.release()

latencies = np.array(latencies)

average = np.mean(latencies)
median = np.percentile(latencies, 50)
p95 = np.percentile(latencies, 95)
minimum = np.min(latencies)
maximum = np.max(latencies)

fps = 1000 / average

total_detections = sum(detection_counts)
average_detections = np.mean(detection_counts)

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"Frames tested        : {len(latencies)}")
print(f"Average latency      : {average:.2f} ms")
print(f"Median / P50         : {median:.2f} ms")
print(f"P95 latency          : {p95:.2f} ms")
print(f"Minimum latency      : {minimum:.2f} ms")
print(f"Maximum latency      : {maximum:.2f} ms")
print(f"Estimated FPS        : {fps:.2f}")
print(f"Total detections     : {total_detections}")
print(f"Avg detections/frame : {average_detections:.2f}")

print("=" * 60)

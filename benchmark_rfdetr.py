import time
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETRNano

IMAGE_DIR = Path("accuracy_test/frames")
NUM_FRAMES = 100

print("=" * 60)
print("RF-DETR NANO LATENCY BENCHMARK")
print("=" * 60)

print("\nLoading RF-DETR Nano...")
model = RFDETRNano()
print("Model loaded.")

images = sorted(IMAGE_DIR.glob("*.jpg"))

print(f"Images found: {len(images)}")

if len(images) < NUM_FRAMES:
    raise RuntimeError("Less than 100 frames found")

# Warmup
print("\nWarming up...")

frame = cv2.imread(str(images[0]))

for _ in range(3):
    model.predict(frame)

print("Warmup complete.")

# Benchmark
latencies = []
detection_counts = []

print("\nTesting 100 frames...\n")

for i, image_path in enumerate(images[:NUM_FRAMES], start=1):

    frame = cv2.imread(str(image_path))

    if frame is None:
        continue

    start = time.perf_counter()

    detections = model.predict(frame)

    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)

    # supervision Detections object
    try:
        count = len(detections.xyxy)
    except Exception:
        count = 0

    detection_counts.append(count)

    if i % 10 == 0:
        print(
            f"Frame {i:3d} | "
            f"Latency: {latency_ms:8.2f} ms | "
            f"Detections: {count}"
        )

# Results
latencies = np.array(latencies)

average = np.mean(latencies)
p50 = np.percentile(latencies, 50)
p95 = np.percentile(latencies, 95)
minimum = np.min(latencies)
maximum = np.max(latencies)

fps = 1000 / average

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"Frames tested        : {len(latencies)}")
print(f"Average latency      : {average:.2f} ms")
print(f"Median / P50         : {p50:.2f} ms")
print(f"P95 latency          : {p95:.2f} ms")
print(f"Minimum latency      : {minimum:.2f} ms")
print(f"Maximum latency      : {maximum:.2f} ms")
print(f"Estimated FPS        : {fps:.2f}")
print(f"Avg detections/frame : {np.mean(detection_counts):.2f}")

print("=" * 60)

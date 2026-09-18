"""Latency benchmark for crowd monitoring — separate from vehicle/smoke benchmarks.

Measures:
  1) YOLO person detection only
  2) Full crowd pipeline (detect + track + flow + panic score)
"""

import argparse
import time

import cv2
import numpy as np

import config
from detectors.crowd_monitoring_core import CrowdFrameProcessor

DEFAULT_VIDEO = "test_videos/Traffic Control CCTV.mp4"
NUM_FRAMES = 100
WARMUP = 5


def run_yolo_only(processor, cap, num_frames):
    latencies = []
    counts = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(WARMUP):
        ret, frame = cap.read()
        if not ret:
            break
        processor.detect_persons(frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        start = time.perf_counter()
        bboxes = processor.detect_persons(frame)
        latencies.append((time.perf_counter() - start) * 1000)
        counts.append(len(bboxes))
    return latencies, counts


def run_full_pipeline(processor_factory, cap, num_frames):
    latencies = []
    counts = []
    panic_scores = []
    processor = processor_factory()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(WARMUP):
        ret, frame = cap.read()
        if not ret:
            break
        processor.process(frame, measure_latency=False)

    processor = processor_factory()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        result = processor.process(frame, measure_latency=True)
        latencies.append(result.latency_ms)
        counts.append(result.crowd_count)
        panic_scores.append(result.panic_score)
    return latencies, counts, panic_scores


def print_stats(title, latencies, counts, extra=None):
    arr = np.array(latencies)
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    print(f"Frames tested        : {len(arr)}")
    print(f"Average latency      : {np.mean(arr):.2f} ms")
    print(f"Median / P50         : {np.percentile(arr, 50):.2f} ms")
    print(f"P95 latency          : {np.percentile(arr, 95):.2f} ms")
    print(f"Min / Max            : {np.min(arr):.2f} / {np.max(arr):.2f} ms")
    print(f"Estimated FPS        : {1000 / np.mean(arr):.2f}")
    print(f"Avg tracked count    : {np.mean(counts):.2f}")
    if extra:
        print(extra)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Crowd monitoring latency benchmark")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--model", default=config.CROWD_MODEL_PATH)
    parser.add_argument("--infer-size", type=int, default=config.CROWD_INFER_SIZE)
    parser.add_argument("--infer-every", type=int, default=config.CROWD_INFER_EVERY)
    parser.add_argument("--flow-every", type=int, default=config.CROWD_FLOW_EVERY)
    args = parser.parse_args()

    print("=" * 60)
    print("CROWD MONITORING LATENCY BENCHMARK")
    print("(independent of benchmark_latency.py / vehicle pipeline)")
    print("=" * 60)

    processor = CrowdFrameProcessor(
        model_path=args.model,
        confidence=config.CROWD_CONFIDENCE,
        infer_every=args.infer_every,
        infer_size=args.infer_size,
        flow_every=args.flow_every,
        sustain_sec=config.CROWD_SUSTAIN_SEC,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")
    print(f"\nVideo: {args.video}")
    print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
    print(f"Model: {args.model} | infer_size={args.infer_size} | infer_every={args.infer_every}")

    y_lat, y_counts = run_yolo_only(processor, cap, args.frames)
    print_stats("YOLO PERSON DETECTION ONLY", y_lat, y_counts)

    def make_processor():
        return CrowdFrameProcessor(
            model_path=args.model,
            confidence=config.CROWD_CONFIDENCE,
            infer_every=args.infer_every,
            infer_size=args.infer_size,
            flow_every=args.flow_every,
            sustain_sec=config.CROWD_SUSTAIN_SEC,
        )

    f_lat, f_counts, panic = run_full_pipeline(make_processor, cap, args.frames)
    print_stats(
        "FULL CROWD PIPELINE (detect + track + flow + panic)",
        f_lat,
        f_counts,
        extra=f"Avg panic score      : {np.mean(panic):.3f}",
    )

    cap.release()


if __name__ == "__main__":
    main()

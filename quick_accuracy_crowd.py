"""Person-detection accuracy for crowd monitoring — separate from quick_accuracy.py.

Uses YOLO-format labels where class id 0 = person (COCO person).
Put images in accuracy_test_crowd/frames and labels in accuracy_test_crowd/labels.

This measures detection quality for the crowd stack, not panic-state ground truth.
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

import config

IMAGE_DIR = Path("accuracy_test_crowd/frames")
LABEL_DIR = Path("accuracy_test_crowd/labels")
PERSON_CLASS_COCO = 0
LABEL_PERSON_CLASS = 0
IOU_THRESHOLD = 0.50


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def load_person_gt(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        values = line.split()
        cls = int(values[0])
        if cls != LABEL_PERSON_CLASS:
            continue
        xc, yc, w, h = map(float, values[1:5])
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return boxes


def main():
    images = sorted(IMAGE_DIR.glob("*.jpg")) + sorted(IMAGE_DIR.glob("*.png"))
    if not images:
        print(f"No images in {IMAGE_DIR}/")
        print("Create accuracy_test_crowd/frames and labels (class 0 = person).")
        return

    print("Loading model for crowd person detection...")
    model = YOLO(config.CROWD_MODEL_PATH)
    print(f"Frames found: {len(images)}")

    tp = fp = fn = 0

    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path.name}")
        results = model.predict(
            source=str(image_path),
            imgsz=config.CROWD_INFER_SIZE,
            conf=config.CROWD_CONFIDENCE,
            classes=[PERSON_CLASS_COCO],
            verbose=False,
        )
        result = results[0]
        h, w = result.orig_shape

        gt_boxes = load_person_gt(LABEL_DIR / f"{image_path.stem}.txt", w, h)
        predictions = []
        if result.boxes is not None:
            for i in range(len(result.boxes)):
                predictions.append(result.boxes.xyxy[i].tolist())

        matched_gt = set()
        for pred_box in predictions:
            best_iou = 0.0
            best_gt = None
            for gt_index, gt_box in enumerate(gt_boxes):
                if gt_index in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_index
            if best_iou >= IOU_THRESHOLD:
                tp += 1
                matched_gt.add(best_gt)
            else:
                fp += 1

        for gt_index in range(len(gt_boxes)):
            if gt_index not in matched_gt:
                fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print()
    print("=" * 70)
    print("CROWD PERSON DETECTION (class 0 labels)")
    print("=" * 70)
    print(f"Frames tested : {len(images)}")
    print(f"TP            : {tp}")
    print(f"FP            : {fp}")
    print(f"FN            : {fn}")
    print(f"Precision     : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall        : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1            : {f1:.4f} ({f1 * 100:.2f}%)")
    print("=" * 70)
    print("NOTE: Panic-state accuracy needs labeled panic segments, not bbox labels alone.")


if __name__ == "__main__":
    main()

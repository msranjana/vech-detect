import cv2
import numpy as np
from pathlib import Path
from rfdetr import RFDETRNano

IMAGE_DIR = Path("accuracy_test/frames")
LABEL_DIR = Path("accuracy_test/labels")

IOU_THRESHOLD = 0.50
CONF_THRESHOLD = 0.25

# RF-DETR / COCO classes
# COCO: car=3, motorcycle=4, bus=6, truck=8
CLASS_MAP = {
    3: 0,  # car
    4: 1,  # motorcycle
    6: 2,  # bus
    8: 3,  # truck
}

CLASS_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
}


def load_labels(label_file):
    labels = []

    if not label_file.exists():
        return labels

    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls, xc, yc, w, h = map(float, parts)

            labels.append({
                "class": int(cls),
                "box": [xc, yc, w, h]
            })

    return labels


def yolo_to_xyxy(box, width, height):
    xc, yc, w, h = box

    x1 = (xc - w / 2) * width
    y1 = (yc - h / 2) * height
    x2 = (xc + w / 2) * width
    y2 = (yc + h / 2) * height

    return [x1, y1, x2, y2]


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


def main():

    print("=" * 60)
    print("RF-DETR NANO PSEUDO-LABEL ACCURACY")
    print("=" * 60)

    print("\nLoading RF-DETR Nano...")
    model = RFDETRNano()
    print("Model loaded.")

    images = sorted(IMAGE_DIR.glob("*.jpg"))

    print(f"Images found: {len(images)}")

    if len(images) == 0:
        raise RuntimeError("No images found.")

    stats = {
        cls: {"TP": 0, "FP": 0, "FN": 0}
        for cls in CLASS_NAMES
    }

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for index, image_path in enumerate(images, start=1):

        frame = cv2.imread(str(image_path))

        if frame is None:
            continue

        height, width = frame.shape[:2]

        # Load YOLO11l pseudo labels
        label_file = LABEL_DIR / f"{image_path.stem}.txt"

        gt_labels = load_labels(label_file)

        gt_boxes = []

        for label in gt_labels:
            box = yolo_to_xyxy(
                label["box"],
                width,
                height
            )

            gt_boxes.append({
                "class": label["class"],
                "box": box,
                "matched": False
            })

        # RF-DETR prediction
        detections = model.predict(frame)

        predictions = []

        for i in range(len(detections.xyxy)):

            confidence = float(detections.confidence[i])

            if confidence < CONF_THRESHOLD:
                continue

            rfdetr_class = int(detections.class_id[i])

            if rfdetr_class not in CLASS_MAP:
                continue

            project_class = CLASS_MAP[rfdetr_class]

            box = detections.xyxy[i].tolist()

            predictions.append({
                "class": project_class,
                "box": box,
                "confidence": confidence
            })

        # Match predictions to pseudo labels
        for prediction in predictions:

            best_iou = 0
            best_gt = None

            for gt in gt_boxes:

                if gt["matched"]:
                    continue

                if gt["class"] != prediction["class"]:
                    continue

                iou = calculate_iou(
                    prediction["box"],
                    gt["box"]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_gt is not None and best_iou >= IOU_THRESHOLD:

                best_gt["matched"] = True

                stats[prediction["class"]]["TP"] += 1
                total_tp += 1

            else:

                stats[prediction["class"]]["FP"] += 1
                total_fp += 1

        # Remaining GT boxes = FN
        for gt in gt_boxes:

            if not gt["matched"]:

                stats[gt["class"]]["FN"] += 1
                total_fn += 1

        if index % 10 == 0:
            print(f"Processed {index}/{len(images)} frames")

    # Overall metrics
    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0 else 0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    print("\n")
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(f"TP        : {total_tp}")
    print(f"FP        : {total_fp}")
    print(f"FN        : {total_fn}")

    print(f"\nPrecision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    print("\n")
    print("=" * 60)
    print("PER CLASS RESULTS")
    print("=" * 60)

    for cls, name in CLASS_NAMES.items():

        tp = stats[cls]["TP"]
        fp = stats[cls]["FP"]
        fn = stats[cls]["FN"]

        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f = 2 * p * r / (p + r) if p + r > 0 else 0

        print(f"\n{name}")
        print(f"  TP        : {tp}")
        print(f"  FP        : {fp}")
        print(f"  FN        : {fn}")
        print(f"  Precision : {p * 100:.2f}%")
        print(f"  Recall    : {r * 100:.2f}%")
        print(f"  F1 Score  : {f * 100:.2f}%")

    print("\n" + "=" * 60)
    print("NOTE: Metrics are agreement with YOLO11l pseudo-labels.")
    print("They are NOT independently validated ground-truth accuracy.")
    print("=" * 60)


if __name__ == "__main__":
    main()

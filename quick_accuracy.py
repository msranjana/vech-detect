from pathlib import Path
from ultralytics import YOLO
import numpy as np

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMAGE_DIR = Path("accuracy_test/frames")
LABEL_DIR = Path("accuracy_test/labels")

MODEL_PATH = "yolov8s.pt"

CONFIDENCE = 0.35
IMAGE_SIZE = 640

# IoU threshold
IOU_THRESHOLD = 0.50

# Our classes
CLASS_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
}

# --------------------------------------------------
# IoU FUNCTION
# --------------------------------------------------

def calculate_iou(box1, box2):
    """
    Boxes are:
    [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading YOLOv8n...")

model = YOLO(MODEL_PATH)

print("Model loaded.")


# --------------------------------------------------
# STORAGE
# --------------------------------------------------

TP = {c: 0 for c in CLASS_NAMES}
FP = {c: 0 for c in CLASS_NAMES}
FN = {c: 0 for c in CLASS_NAMES}

total_frames = 0


# --------------------------------------------------
# PROCESS FRAMES
# --------------------------------------------------

images = sorted(IMAGE_DIR.glob("*.jpg"))

print("Frames found:", len(images))

for frame_number, image_path in enumerate(images, start=1):

    print(f"[{frame_number}/{len(images)}] {image_path.name}")

    # Read pseudo ground truth
    label_file = LABEL_DIR / f"{image_path.stem}.txt"

    ground_truth = []

    if label_file.exists():

        for line in label_file.read_text().splitlines():

            if not line.strip():
                continue

            values = line.split()

            cls = int(values[0])

            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

            ground_truth.append(
                (
                    cls,
                    x_center,
                    y_center,
                    width,
                    height,
                )
            )

    # YOLOv8n prediction
    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        classes=[2, 3, 5, 7],
        verbose=False,
    )

    result = results[0]

    predictions = []

    if result.boxes is not None:

        for i in range(len(result.boxes)):

            coco_class = int(result.boxes.cls[i].item())

            # Convert COCO → our classes
            mapping = {
                2: 0,  # car
                3: 1,  # motorcycle
                5: 2,  # bus
                7: 3,  # truck
            }

            if coco_class not in mapping:
                continue

            cls = mapping[coco_class]

            xyxy = result.boxes.xyxy[i].tolist()

            predictions.append(
                (
                    cls,
                    xyxy
                )
            )

    # --------------------------------------------------
    # Convert GT normalized boxes to pixels
    # --------------------------------------------------

    height, width = result.orig_shape

    gt_boxes = []

    for cls, xc, yc, w, h in ground_truth:

        x1 = (xc - w / 2) * width
        y1 = (yc - h / 2) * height
        x2 = (xc + w / 2) * width
        y2 = (yc + h / 2) * height

        gt_boxes.append(
            (
                cls,
                [x1, y1, x2, y2]
            )
        )

    # --------------------------------------------------
    # MATCH PREDICTIONS TO GT
    # --------------------------------------------------

    matched_gt = set()

    for pred_cls, pred_box in predictions:

        best_iou = 0
        best_gt = None

        for gt_index, (gt_cls, gt_box) in enumerate(gt_boxes):

            if gt_index in matched_gt:
                continue

            if pred_cls != gt_cls:
                continue

            iou = calculate_iou(pred_box, gt_box)

            if iou > best_iou:

                best_iou = iou
                best_gt = gt_index

        if best_iou >= IOU_THRESHOLD:

            TP[pred_cls] += 1
            matched_gt.add(best_gt)

        else:

            FP[pred_cls] += 1

    # Unmatched GT = FN

    for gt_index, (gt_cls, _) in enumerate(gt_boxes):

        if gt_index not in matched_gt:

            FN[gt_cls] += 1

    total_frames += 1


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print()
print("=" * 70)
print("YOLOv8n vs YOLO11l PSEUDO-LABELS")
print("=" * 70)

total_tp = sum(TP.values())
total_fp = sum(FP.values())
total_fn = sum(FN.values())

precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp > 0
    else 0
)

recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn > 0
    else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0
    else 0
)

print(f"Frames tested : {total_frames}")

print()
print("Overall:")
print(f"TP        : {total_tp}")
print(f"FP        : {total_fp}")
print(f"FN        : {total_fn}")
print(f"Precision : {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score  : {f1:.4f} ({f1*100:.2f}%)")

print()
print("Per class:")
print("-" * 70)

for cls, name in CLASS_NAMES.items():

    p = (
        TP[cls] / (TP[cls] + FP[cls])
        if TP[cls] + FP[cls] > 0
        else 0
    )

    r = (
        TP[cls] / (TP[cls] + FN[cls])
        if TP[cls] + FN[cls] > 0
        else 0
    )

    f = (
        2 * p * r / (p + r)
        if p + r > 0
        else 0
    )

    print(
        f"{name:12s} "
        f"TP={TP[cls]:4d} "
        f"FP={FP[cls]:4d} "
        f"FN={FN[cls]:4d} "
        f"P={p*100:6.2f}% "
        f"R={r*100:6.2f}% "
        f"F1={f*100:6.2f}%"
    )

print("=" * 70)
print()
print("NOTE:")
print("These metrics measure agreement with YOLO11l pseudo-labels.")
print("They are NOT independent ground-truth accuracy.")
